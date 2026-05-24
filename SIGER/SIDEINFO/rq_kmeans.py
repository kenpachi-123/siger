import argparse
import os
import time
from collections import Counter

import numpy as np
from k_means_constrained import KMeansConstrained
import torch



def set_global_seed(seed: int):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(True, warn_only=True)

# 实验开关：
# - "none": 不做残差归一化
# - "full": 训练/编码/解码全流程使用残差归一化
# - "codebook_only": 训练时不做残差归一化，但导出的 codebook 按残差归一化格式保存
RQ1_RESIDUAL_NORMALIZATION_MODE = "full"

# 残差归一化与均衡KMeans结合的RQ-Kmeans方法+碰撞消解，确保每个物品都有唯一的编码组合
class OptimizedRQKMeans:
    def __init__(self, n_stages=2, n_clusters=256, max_iter=200, random_state=42,
                 use_gpu=True, gpu_id=0, batch_size=1000, balance_tolerance=1.5,
                 float_precision=64,
                 residual_normalization_mode=RQ1_RESIDUAL_NORMALIZATION_MODE):
        """
        优化的RQ-Kmeans实现（方法一：均衡KMeans，限制簇大小上限降低碰撞率）

        Args:
            n_stages:          量化阶段数
            n_clusters:        每个阶段的聚类数（码本大小）
            max_iter:          kmeans最大迭代次数
            random_state:      随机种子
            use_gpu:           是否使用GPU（需要torch CUDA支持）
            batch_size:        批处理大小
            balance_tolerance: 簇大小上限容忍倍数（建议 1.2~2.0）。
                               每簇最多容纳 ceil(n/K) * balance_tolerance 个样本。
                               1.0 = 完全均匀；值越大约束越宽松、重构误差越小。
            float_precision:   数值精度，32 或 64
            residual_normalization_mode:
                               "none" / "full" / "codebook_only"
        """
        self.n_stages = n_stages
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.balance_tolerance = balance_tolerance
        self.float_precision = int(float_precision)
        self.residual_normalization_mode = str(residual_normalization_mode).lower()
        if self.residual_normalization_mode not in {"none", "full", "codebook_only"}:
            raise ValueError(
                "residual_normalization_mode must be one of: none, full, codebook_only"
            )
        if self.float_precision == 32:
            self.np_dtype = np.float32
            self.torch_dtype = torch.float32
        elif self.float_precision == 64:
            self.np_dtype = np.float64
            self.torch_dtype = torch.float64
        else:
            raise ValueError(f"float_precision must be 32 or 64, got {self.float_precision}")

        self.codebooks = []
        self.is_fitted = False

        # 统计信息
        self.stage_usage_stats = []  # 每个阶段的码本使用统计
        self.collision_stats = {}  # 碰撞统计

        # 方法五：残差归一化 scale 列表（共 n_stages-1 个）
        self.residual_scales = []

        # 设备选择
        self.device = torch.device(f'cuda:{self.gpu_id}' if self.use_gpu else 'cpu')

    def _tensor_to_numpy(self, tensor):
        return tensor.detach().cpu().numpy().astype(self.np_dtype, copy=False)

    def _use_full_residual_normalization(self):
        return self.residual_normalization_mode == "full"

    def _export_residual_normalized_codebooks(self):
        return self.residual_normalization_mode in {"full", "codebook_only"}

    def _get_cumulative_scales(self, include_codebook_only=False):
        cumulative_scales = [1.0]
        should_apply = self._use_full_residual_normalization() or (
            include_codebook_only and self.residual_normalization_mode == "codebook_only"
        )
        if should_apply:
            for scale in self.residual_scales:
                cumulative_scales.append(cumulative_scales[-1] * scale)
        else:
            cumulative_scales.extend([1.0] * (self.n_stages - 1))
        return cumulative_scales

    def fit(self, X):
        """
        训练RQ-Kmeans

        Args:
            X: 输入向量，shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=self.np_dtype)
        n_samples, _ = X.shape

        self.codebooks = []
        self.stage_usage_stats = []
        self.residual_scales = []

        residual = torch.from_numpy(X).to(device=self.device, dtype=self.torch_dtype)

        for stage in range(self.n_stages):
            print(f"Training stage {stage + 1}/{self.n_stages}...")

            # KMeansConstrained 需要 NumPy 输入
            residual_np = self._tensor_to_numpy(residual)

            # 方法一：使用 KMeansConstrained 限制每簇最大容量，
            # 从源头消除热门簇过大导致的码本碰撞。
            ideal_size = int(np.ceil(n_samples / self.n_clusters))
            size_max = int(ideal_size * self.balance_tolerance)
            # size_min 设为 1，保证不出现空簇
            size_min = 1

            print(f"  KMeansConstrained: size_min={size_min}, "
                  f"size_max={size_max} "
                  f"(ideal={ideal_size}, tol={self.balance_tolerance})")

            kmeans = KMeansConstrained(
                n_clusters=self.n_clusters,
                size_min=size_min,
                size_max=size_max,
                max_iter=self.max_iter,
                random_state=self.random_state + stage,
                n_init=3,
            )

            cluster_labels = kmeans.fit_predict(residual_np)
            codebook = kmeans.cluster_centers_.astype(self.np_dtype, copy=False)

            # 存储码本
            self.codebooks.append(codebook)

            # 统计训练时的码本使用情况
            usage_counter = Counter(cluster_labels)
            stage_stats = {
                'total_codes': self.n_clusters,
                'used_codes': len(usage_counter),
                'usage_ratio': len(usage_counter) / self.n_clusters,
                'usage_distribution': dict(usage_counter),
                'max_usage': max(usage_counter.values()),
                'min_usage': min(usage_counter.values()),
                'avg_usage': np.mean(list(usage_counter.values())),
                'std_usage': np.std(list(usage_counter.values()))
            }
            self.stage_usage_stats.append(stage_stats)

            # 计算量化向量并更新残差
            codebook_tensor = torch.from_numpy(codebook).to(device=self.device, dtype=self.torch_dtype)
            labels_tensor = torch.from_numpy(cluster_labels).to(device=self.device, dtype=torch.long)
            residual = residual - codebook_tensor[labels_tensor]

            # ── 方法五：自适应残差归一化 ────────────────────────────────────
            # 将残差缩放到单位量级后再送入下一阶段，防止后期残差趋零导致
            # 多个 item 被映射到同一码本中心。仅在非末尾阶段执行。
            if stage < self.n_stages - 1 and self.residual_normalization_mode in {"full", "codebook_only"}:
                scale = float(torch.linalg.vector_norm(residual, dim=1).mean().item())
                scale = max(scale, 1e-8)
                self.residual_scales.append(scale)
                if self._use_full_residual_normalization():
                    residual = residual / scale
                print(
                    f"  Stage {stage + 1} → residual scale: {scale:.6f}"
                    f"{'  (normalized for stage ' + str(stage + 2) + ')' if self._use_full_residual_normalization() else '  (recorded for codebook export only)'}"
                )

        self.is_fitted = True
        print("Training completed!")

    def encode_batch(self, X, collect_stats=True, resolve_collisions=False):
        """
        批量编码向量

        Args:
            X: 输入向量，shape (n_samples, n_features)
            collect_stats: 是否收集统计信息

        Returns:
            codes: 量化码，shape (n_samples, n_stages)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before encoding")

        X = np.asarray(X, dtype=self.np_dtype)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        codes = np.zeros((n_samples, self.n_stages), dtype=np.int32)
        residual = torch.from_numpy(X).to(device=self.device, dtype=self.torch_dtype)

        # 统计信息收集
        stage_code_usage = []

        for stage in range(self.n_stages):
            codebook_tensor = torch.from_numpy(self.codebooks[stage]).to(
                device=self.device, dtype=self.torch_dtype
            )
            distances = torch.cdist(residual, codebook_tensor)
            cluster_ids_t = torch.argmin(distances, dim=1)
            cluster_ids = cluster_ids_t.detach().cpu().numpy().astype(np.int32, copy=False)

            codes[:, stage] = cluster_ids

            # 收集当前阶段的码本使用统计
            if collect_stats:
                stage_code_usage.append(cluster_ids)

            # 更新残差
            codebook_tensor = torch.from_numpy(self.codebooks[stage]).to(
                device=self.device, dtype=self.torch_dtype
            )
            residual = residual - codebook_tensor[cluster_ids_t]

            # 方法五：与 fit 保持一致，按保存的 scale 归一化残差
            if self._use_full_residual_normalization() and stage < self.n_stages - 1:
                residual = residual / self.residual_scales[stage]

        # 碰撞解决：将具有相同编码组合的 item 重新分配到次优空闲编码
        if resolve_collisions:
            codes = self._resolve_collisions(X, codes)

        # 更新统计信息（内部仍使用 0-based 计算，保证与码本索引一致）
        if collect_stats:
            self._update_encoding_stats(codes, stage_code_usage)

        return codes

    def _resolve_collisions(self, X, codes):
        """
        检测所有碰撞（多个 item 具有相同编码组合），
        对碰撞 item 强制重新分配到距离次优的空闲编码组合。

        策略：
        1. 在每个阶段按当前残差计算 item 到所有码本中心的距离
        2. 按总量化误差排序，保留误差最小的 item 不动
        3. 对其余 item，遍历每个阶段找最近的可用替代编码
        4. 选择额外代价最小的（stage, alt_code）执行替换
        """
        n_samples = codes.shape[0]
        new_codes = codes.copy()

        # ── 1. 缓存每阶段的距离矩阵 (n_samples, n_clusters) ──
        stage_distances = []
        residual = torch.from_numpy(np.asarray(X, dtype=self.np_dtype)).to(
            device=self.device, dtype=self.torch_dtype
        )
        for stage in range(self.n_stages):
            codebook_tensor = torch.from_numpy(self.codebooks[stage]).to(
                device=self.device, dtype=self.torch_dtype
            )
            dists = self._tensor_to_numpy(torch.cdist(residual, codebook_tensor))
            stage_distances.append(dists)
            labels_tensor = torch.from_numpy(new_codes[:, stage]).to(device=self.device, dtype=torch.long)
            residual = residual - codebook_tensor[labels_tensor]
            if self._use_full_residual_normalization() and stage < self.n_stages - 1:
                residual = residual / self.residual_scales[stage]

        # ── 2. 识别碰撞组 ──
        code_tuples = [tuple(new_codes[i]) for i in range(n_samples)]
        combo_counter = Counter(code_tuples)
        collision_combos = {c for c, cnt in combo_counter.items() if cnt > 1}

        if not collision_combos:
            print("  碰撞解决：无碰撞，跳过。")
            return new_codes

        collision_groups = {c: [] for c in collision_combos}
        for i, c in enumerate(code_tuples):
            if c in collision_combos:
                collision_groups[c].append(i)

        total_to_reassign = sum(len(g) - 1 for g in collision_groups.values())
        print(f"  碰撞解决：发现 {len(collision_groups)} 组碰撞，"
              f"共 {total_to_reassign} 个 item 需重新分配")

        # ── 3. 已占用的编码组合（包含所有 item，不仅是碰撞组） ──
        used_combos = set(code_tuples)

        # ── 4. 贪心逐组解决 ──
        resolved = 0
        for combo, indices in collision_groups.items():
            # 按总量化误差排序，误差最小的保留原编码
            errs = [sum(stage_distances[s][idx, new_codes[idx, s]]
                        for s in range(self.n_stages))
                    for idx in indices]
            order = [indices[j] for j in np.argsort(errs)]

            for idx in order[1:]:
                best_alt = None
                best_extra = float('inf')

                for stage in range(self.n_stages):
                    cur = new_codes[idx, stage]
                    dists = stage_distances[stage][idx]
                    ranking = np.argsort(dists)

                    for alt in ranking:
                        if alt == cur:
                            continue
                        cand = list(new_codes[idx])
                        cand[stage] = int(alt)
                        cand_t = tuple(cand)
                        if cand_t not in used_combos:
                            extra = dists[alt] - dists[cur]
                            if extra < best_extra:
                                best_extra = extra
                                best_alt = (stage, int(alt), cand_t)
                            break  # 该阶段的最优替代已找到，换下一阶段

                if best_alt is not None:
                    s_pick, c_pick, new_combo = best_alt
                    new_codes[idx, s_pick] = c_pick
                    used_combos.add(new_combo)
                    resolved += 1

        print(f"  碰撞解决：成功重新分配 {resolved}/{total_to_reassign}")
        return new_codes

    def _update_encoding_stats(self, codes, stage_code_usage):
        """更新编码统计信息"""
        n_samples = codes.shape[0]

        # 计算码本组合碰撞率
        code_combinations = []
        for i in range(n_samples):
            combo = tuple(codes[i])
            code_combinations.append(combo)

        combo_counter = Counter(code_combinations)
        unique_combinations = len(combo_counter)
        collision_rate = 1.0 - (unique_combinations / n_samples)

        # 更新全局统计
        self.collision_stats = {
            'total_samples': n_samples,
            'unique_combinations': unique_combinations,
            'collision_rate': collision_rate,
            'most_common_combinations': combo_counter.most_common(10),
            'theoretical_max_combinations': self.get_codebook_size(),
            'combination_utilization': unique_combinations / self.get_codebook_size()
        }

        # 更新每个阶段的使用统计
        for stage, cluster_ids in enumerate(stage_code_usage):
            usage_counter = Counter(cluster_ids)
            self.stage_usage_stats[stage].update({
                'encoding_used_codes': len(usage_counter),
                'encoding_usage_ratio': len(usage_counter) / self.n_clusters,
                'encoding_usage_distribution': dict(usage_counter)
            })

    def get_collision_stats(self):
        """获取碰撞统计信息"""
        return self.collision_stats

    def get_codebook_usage_stats(self):
        """获取码本使用效率统计"""
        return self.stage_usage_stats

    def print_detailed_stats(self):
        """打印详细的统计信息"""
        print("\n" + "=" * 60)
        print("RQ-KMeans 详细统计报告")
        print("=" * 60)

        # 基本信息
        print(f"量化阶段数: {self.n_stages}")
        print(f"每阶段聚类数: {self.n_clusters}")
        print(f"理论最大组合数: {self.get_codebook_size():,}")
        print(f"残差归一化模式: {self.residual_normalization_mode}")

        # 残差归一化 scale（方法五）
        if self.residual_scales:
            print("\n残差归一化 Scale（方法五）:")
            print("-" * 40)
            cumulative = 1.0
            for i, s in enumerate(self.residual_scales):
                cumulative *= s
                print(f"  阶段 {i + 1} → 阶段 {i + 2}: scale={s:.6f}  "
                      f"（累积还原系数={cumulative:.6f}）")

        # 码本使用效率
        print("\n码本使用效率分析:")
        print("-" * 40)
        for stage, stats in enumerate(self.stage_usage_stats):
            print(f"阶段 {stage + 1}:")
            print(f"  训练时使用的码本数: {stats['used_codes']}/{stats['total_codes']} "
                  f"({stats['usage_ratio']:.2%})")

            if 'encoding_used_codes' in stats:
                print(f"  编码时使用的码本数: {stats['encoding_used_codes']}/{stats['total_codes']} "
                      f"({stats['encoding_usage_ratio']:.2%})")

            print(f"  使用次数统计: 最大={stats['max_usage']}, 最小={stats['min_usage']}, "
                  f"平均={stats['avg_usage']:.1f}, 标准差={stats['std_usage']:.1f}")

        # 碰撞分析
        if self.collision_stats:
            print("\n碰撞分析:")
            print("-" * 40)
            print(f"总样本数: {self.collision_stats['total_samples']:,}")
            print(f"唯一组合数: {self.collision_stats['unique_combinations']:,}")
            print(f"碰撞率: {self.collision_stats['collision_rate']:.2%}")
            print(f"组合空间利用率: {self.collision_stats['combination_utilization']:.2%}")

            print("\n最常见的组合 (前5个):")
            for i, (combo, count) in enumerate(self.collision_stats['most_common_combinations'][:5]):
                print(f"  {i + 1}. {combo}: {count} 次 ({count / self.collision_stats['total_samples']:.2%})")

    def analyze_codebook_balance(self):
        """分析码本负载均衡情况"""
        balance_analysis = {}

        for stage, stats in enumerate(self.stage_usage_stats):
            if 'encoding_usage_distribution' in stats:
                usage_counts = list(stats['encoding_usage_distribution'].values())
            else:
                usage_counts = list(stats['usage_distribution'].values())

            # 计算基尼系数衡量不平衡程度
            sorted_counts = sorted(usage_counts)
            n = len(sorted_counts)
            cumsum = np.cumsum(sorted_counts)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * cumsum[-1]) - (n + 1) / n

            # 计算变异系数
            cv = np.std(usage_counts) / np.mean(usage_counts)

            balance_analysis[f'stage_{stage + 1}'] = {
                'gini_coefficient': gini,
                'coefficient_of_variation': cv,
                'balance_score': 1.0 - gini,  # 平衡分数，越接近1越平衡
                'interpretation': 'balanced' if gini < 0.2 else 'moderately_unbalanced' if gini < 0.4 else 'highly_unbalanced'
            }

        return balance_analysis

    def encode(self, x):
        """单个向量编码（兼容原接口）"""
        codes = self.encode_batch(x.reshape(1, -1) if x.ndim == 1 else x, collect_stats=False)
        return codes[0].tolist() if codes.shape[0] == 1 else codes

    def decode_batch(self, codes):
        """
        批量解码量化码

        Args:
            codes: 量化码，shape (n_samples, n_stages)

        Returns:
            reconstructed: 重构向量，shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")

        codes = np.asarray(codes)
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)

        n_samples = codes.shape[0]
        n_features = self.codebooks[0].shape[1]

        # 方法五：阶段 k 的码本向量处于经过前 k 次归一化的残差空间中，
        # decode 时需乘以累积还原系数才能回到原始空间：
        #   阶段 0: 系数 = 1.0
        #   阶段 k: 系数 = residual_scales[0] * ... * residual_scales[k-1]
        cumulative_scales = self._get_cumulative_scales()

        codes_tensor = torch.from_numpy(codes.astype(np.int64, copy=False)).to(self.device)
        reconstructed = torch.zeros(
            (n_samples, n_features), device=self.device, dtype=self.torch_dtype
        )
        for stage in range(self.n_stages):
            codebook_tensor = torch.from_numpy(self.codebooks[stage]).to(
                device=self.device, dtype=self.torch_dtype
            )
            reconstructed += codebook_tensor[codes_tensor[:, stage]] * cumulative_scales[stage]
        return self._tensor_to_numpy(reconstructed)

    def decode(self, codes):
        """单个码解码（兼容原接口）"""
        if isinstance(codes, list):
            codes = np.array(codes)
        result = self.decode_batch(codes.reshape(1, -1) if codes.ndim == 1 else codes)
        return result[0] if result.shape[0] == 1 else result

    def get_codebook_size(self):
        """返回总码本大小"""
        return self.n_clusters ** self.n_stages

    def get_compression_ratio(self, original_dim):
        """计算压缩比"""
        original_bits = original_dim * self.float_precision
        compressed_bits = self.n_stages * np.log2(self.n_clusters)
        return original_bits / compressed_bits

    def save_codebooks(self, path):
        codebooks = [np.asarray(cb, dtype=self.np_dtype).copy() for cb in self.codebooks]
        if self.residual_normalization_mode == "codebook_only":
            cumulative_scales = self._get_cumulative_scales(include_codebook_only=True)
            codebooks = [
                cb / cumulative_scales[stage]
                for stage, cb in enumerate(codebooks)
            ]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            codebooks=np.array(codebooks, dtype=object),
            float_precision=np.array(self.float_precision, dtype=np.int32),
            residual_scales=np.array(self.residual_scales, dtype=self.np_dtype),
            residual_normalization_mode=np.array(self.residual_normalization_mode),
            n_stages=np.array(self.n_stages, dtype=np.int32),
            n_clusters=np.array(self.n_clusters, dtype=np.int32),
        )
        print(f"Codebooks saved to {path}")


# 保持原类名的兼容性
class RQKMeans(OptimizedRQKMeans):
    """向后兼容的原类名"""
    pass


def run_rq_pipeline(args):
    set_global_seed(args.random_state)
    #args.use_gpu=False  # 强制禁用 GPU，确保在 CPU 环境下测试
    X = np.load(args.embedding_path)
    X = np.asarray(X, dtype=np.float64 if args.float_precision == 64 else np.float32)
    print(f"Loaded embedding shape: {X.shape}")

    n_samples, n_features = X.shape
    residual_norm_desc = {
        "none": "无残差归一化",
        "full": "方法五(残差归一化)",
        "codebook_only": "仅导出码本时使用残差归一化格式",
    }[RQ1_RESIDUAL_NORMALIZATION_MODE]
    print(f"Testing RQ-KMeans: 方法一(均衡KMeans) + {residual_norm_desc} + 碰撞消解...")
    print(
        f"Config: float_precision={args.float_precision}, use_gpu={args.use_gpu}, "
        f"gpu_id={args.gpu_id}, random_state={args.random_state}, "
        f"residual_normalization_mode={RQ1_RESIDUAL_NORMALIZATION_MODE}"
    )

    start_time = time.time()
    rq_opt = OptimizedRQKMeans(
        n_stages=args.n_stages,
        n_clusters=args.n_clusters,
        max_iter=args.max_iter,
        random_state=args.random_state,
        use_gpu=args.use_gpu,
        gpu_id=args.gpu_id,
        balance_tolerance=args.balance_tolerance,
        float_precision=args.float_precision,
        residual_normalization_mode=RQ1_RESIDUAL_NORMALIZATION_MODE,
    )
    rq_opt.fit(X)
    fit_time = time.time() - start_time
    print(f"Fit time (balanced, mode={RQ1_RESIDUAL_NORMALIZATION_MODE}): {fit_time:.2f}s")

    start_time = time.time()
    codes_batch = rq_opt.encode_batch(
        X,
        collect_stats=True,
        resolve_collisions=not args.disable_collision_resolution,
    )
    encode_time = time.time() - start_time
    print(f"Batch encode time ({n_samples} vectors): {encode_time:.4f}s")
    print(f"Average encode time per vector: {encode_time / n_samples * 1000:.4f}ms")

    codes_to_save = codes_batch + 1 if args.code_start_index == 1 else codes_batch
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    np.save(args.output_path, codes_to_save)
    print(
        f"Codes saved to {args.output_path}, shape={codes_to_save.shape}, "
        f"range=[{codes_to_save.min()}, {codes_to_save.max()}]"
    )

    if args.codebooks_output:
        rq_opt.save_codebooks(args.codebooks_output)

    start_time = time.time()
    reconstructed_batch = rq_opt.decode_batch(codes_batch)
    decode_time = time.time() - start_time
    print(f"Batch decode time: {decode_time:.4f}s")

    error = np.mean(np.linalg.norm(X - reconstructed_batch, axis=1))
    print(f"Average reconstruction error: {error:.4f}")
    print(f"Compression ratio: {rq_opt.get_compression_ratio(n_features):.2f}x")

    rq_opt.print_detailed_stats()

    balance_analysis = rq_opt.analyze_codebook_balance()
    print("\n码本负载均衡分析:")
    print("-" * 40)
    for stage, analysis in balance_analysis.items():
        print(f"{stage}: 基尼系数={analysis['gini_coefficient']:.3f}, "
              f"平衡分数={analysis['balance_score']:.3f}, "
              f"状态={analysis['interpretation']}")


def build_parser():
    parser = argparse.ArgumentParser(description="基于 item embedding 训练 RQ-KMeans 并导出量化码")
    parser.add_argument("--embedding_path", type=str, required=True, help="输入 item embedding 的 .npy 路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出量化码 .npy 路径")
    parser.add_argument("--codebooks_output", type=str, default=None, help="可选：输出码本 .npz 路径")
    parser.add_argument("--n_stages", type=int, default=4, help="RQ 阶段数")
    parser.add_argument("--n_clusters", type=int, default=256, help="每阶段聚类数")
    parser.add_argument("--max_iter", type=int, default=200, help="KMeans 最大迭代次数")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--balance_tolerance", type=float, default=1.5, help="均衡 KMeans 的簇容量容忍倍数")
    parser.add_argument("--float_precision", type=int, choices=[32, 64], default=32,
                        help="NumPy/Torch 数值精度")
    parser.add_argument("--gpu_id", type=int, default=0, help="使用的 CUDA 设备编号")
    parser.add_argument("--code_start_index", type=int, choices=[0, 1], default=0,
                        help="保存量化码时的起始索引；现有 npy 格式使用 0")
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true", help="启用 GPU 编码/残差计算")
    parser.add_argument("--no_gpu", dest="use_gpu", action="store_false", help="禁用 GPU 编码/残差计算")
    parser.add_argument("--disable_collision_resolution", action="store_true", help="关闭碰撞消解")
    parser.set_defaults(use_gpu=True)
    return parser


if __name__ == "__main__":
    run_rq_pipeline(build_parser().parse_args())
