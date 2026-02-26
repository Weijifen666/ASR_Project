from utils import *          # 提供 read_feats_and_targets / get_feats / read_all_data 等
import numpy as np
import scipy.cluster.vq as vq

# =========================
# 可调超参数
# =========================
num_gaussian = 5      # 每个 GMM 的高斯分量数 K
num_iterations = 5    # EM 迭代次数

# 11 类（与你的 text 标注一致：0 用 Z 表示，o 用 O 表示）
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def _logsumexp(a, axis=None, keepdims=False):
    """
    数值稳定版 log-sum-exp：
        log(sum(exp(a))) = m + log(sum(exp(a - m)))
    用于避免 exp 下溢/上溢。

    参数:
      a: ndarray
      axis: 沿哪个维度求和
      keepdims: 是否保留维度

    返回:
      logsumexp 的结果
    """
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)

    # 如果 a_max 是 -inf（例如所有元素都是 -inf），避免出现 nan
    is_finite = np.isfinite(a_max)
    a_max = np.where(is_finite, a_max, 0.0)

    s = np.sum(np.exp(a - a_max), axis=axis, keepdims=True)
    out = a_max + np.log(s + 1e-300)  # +1e-300 防止 log(0)

    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


class GMM:
    """
    Full-covariance GMM (全协方差高斯混合模型), 用 EM 训练。

    参数:
      dim = D: 特征维度（本实验为 39）
      K: 高斯分量数
      eps: 协方差正则项，避免奇异/不可逆

    内部参数:
      pi:    (K,)         混合权重
      mu:    (K, D)       均值
      sigma: (K, D, D)    协方差（全协方差）
    """
    def __init__(self, D, K=5, eps=1e-6):
        assert D > 0
        self.dim = D
        self.K = K
        self.eps = float(eps)

        # 用 KMeans 在全体训练帧上做初始化（注意：这是 scaffold 给定的设计）
        self.mu, self.sigma, self.pi = self.kmeans_initial()

        # 转成 ndarray，统一 shape
        self.mu = np.asarray(self.mu, dtype=np.float64)            # (K, D)
        self.sigma = np.asarray(self.sigma, dtype=np.float64)      # (K, D, D)
        self.pi = np.asarray(self.pi, dtype=np.float64)            # (K,)

        # 保底：协方差正则 + 权重归一化
        self._regularize_all_sigmas()
        self._normalize_pi()

    def _normalize_pi(self):
        """
        保证 pi > 0 且和为 1（避免出现 log(0) 或数值漂移）
        """
        self.pi = np.maximum(self.pi, 1e-12)
        self.pi = self.pi / np.sum(self.pi)

    def _regularize_sigma(self, sigma):
        """
        给协方差加 eps * I，保证正定/可逆。
        注意：不能用 sigma + 0.0001（给所有元素加常数会破坏结构）
        """
        return sigma + self.eps * np.eye(self.dim, dtype=np.float64)

    def _regularize_all_sigmas(self):
        """
        对所有分量的协方差做正则。
        """
        for k in range(self.K):
            self.sigma[k] = self._regularize_sigma(self.sigma[k])

    def kmeans_initial(self):
        """
        KMeans 初始化：
          - 读取所有训练帧 data: (N, D)
          - KMeans 聚类得到 labels
          - 每个簇的均值作为 mu[k]
          - 每个簇的协方差作为 sigma[k]
          - 每个簇占比作为 pi[k]

        返回:
          mu: list of (D,)
          sigma: list of (D,D)
          pi: (K,)
        """
        #0.创建空数
        mu = []  #均值
        sigma = []  #方差

        #1.读取数据，再将数据转化为ndarray数组，便于后续运算
        data = read_all_data('train/feats.scp')  # 期望 shape: (N, D)
        data = np.asarray(data, dtype=np.float64)

        #2.检查数据数组的维度是否正确（N，39）
        if data.ndim != 2 or data.shape[1] != self.dim:
            #ndarray数组.+属性：data.ndim表示数组维度数，data.shape[1]为维度索引，值应为39
            raise ValueError(f"Expected data shape (N,{self.dim}), got {data.shape}")
        '''
        raise是Python的关键字，意思是"主动抛出异常"。
        raise的语法:
        raise异常类型("错误信息")
        '''

        # 3.KMeans 聚类得到标签
        # scipy 的 kmeans2：minit="points" 表示从数据点中选初始中心
        '''
        返回的labels随机示例：
        labels = [
            2,  # 第1帧属于第2类（低音）
            0,  # 第2帧属于第0类（高音）
            2,  # 第3帧属于第2类（低音）
            1,  # 第4帧属于第1类（中音）
            3,  # 第5帧属于第3类
            2,  # 第6帧属于第2类
            0,  # 第7帧属于第0类
            ... 一共10000个标签
            ]
        '''
        (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)
        #data-要聚类的数据；
        # self.K-要聚成K类;
        # minit="points"-初始化方法;
        # minit 是 "method of initialization" 的缩写
        # 意思是：如何选择初始的聚类中
        # "points"的具体含义：
        # 从数据中随机选择K个点作为初始聚类中心
        # 其他可选值：
        # minit = "random"  # 随机生成K个点（不一定在数据中）
        # minit = "++"  # 用KMeans++算法（更聪明的方式）
        # minit = "matrix"  # 用户提供初始中心矩阵

        #4. 按簇收集样本
        clusters = [[] for _ in range(self.K)]
        for (l, d) in zip(labels, data):
            clusters[int(l)].append(d)  #数组名称.append(),表示在这个数组后面添加东西

        #5.空簇保护：若某簇为空，就随机塞一个点进去
        rng = np.random.default_rng(0)  # 这里的 0 就是随机种子，表示创建一个随机数生成器
        for i in range(self.K):
            if len(clusters[i]) == 0:
                clusters[i].append(data[rng.integers(0, len(data))])
                #rng.integers(0,len(data))
                # 第一个参数 0：起始值（包含）
                # 第二个参数 1en(data)：结束值（不包含）
                # 所以生成的是 0 到 数据总数的值 之间的整数

        #6.计算每个簇的均值和协方差
        #6.1 计算均值
        for cluster in clusters:  #直接依次遍历元素，每取一次就对其执行其后的代码
            cluster = np.asarray(cluster, dtype=np.float64)
            mu.append(np.mean(cluster, axis=0)) #np.mean（）对一个数组求均值，方向由axis的值确定
            # 二维数组有两个轴：
            # axis=0：垂直方向（行，从上到下）
            # axis=1：水平方向（列，从左到右）
            ''' 
            axis = 1(水平方向)
            ←---------→
            ┌─────────────┐
            │ [1, 2]      │ ↓
            │ [3, 4]      │ axis = 0(垂直方向)
            │ [5, 6]      │ ↓
            └─────────────┘
            '''
        #6.2 计算协方差
            # 如果簇里只有 1 个点，np.cov 会不稳定，这里用全局方差的对角阵兜底
            if cluster.shape[0] == 1:
                v = np.var(data, axis=0) + 1e-3   #np.var()计算方差（variance），并在v这个39维数组当中每个元素都加1e-3
                sigma.append(np.diag(v))    #np.diag()从向量创建对角矩阵
                #最终结果	sigma 是一个长度为K的列表，每个元素是协方差矩阵
            else:
                # bias=True: 除以 N（而不是 N-1），更像极大似然估计
                cov = np.cov(cluster, rowvar=0, bias=True)  #np.cov() 计算协方差矩阵（covariance matrix）
                # rowvar=0（代码中的设置）：
                # 表示：每一列代表一个变量（特征），每一行代表一个观测（样本）
                # bias=True：除以 N（样本数）- 极大似然估计
                #协方差 = sum((x - mean)(y - mean)) / N
                # bias=False：除以 N-1 - 无偏估计（默认）
                #协方差 = sum((x - mean)(y - mean)) / (N - 1)
                sigma.append(cov)
                # 对于39维的MFCC特征，协方差矩阵是一个39×39的矩阵
                # 对角线元素：每个特征自己的方差
                # 非对角线元素：不同特征之间的协方差（相关性）
                '''
                cov_matrix = [
                    [var_1, cov_12, cov_13, ..., cov_1_39],
                    [cov_21, var_2, cov_23, ..., cov_2_39],
                    [cov_31, cov_32, var_3, ..., cov_3_39],
                    ...,
                    [cov_39_1, cov_39_2, cov_39_3, ..., var_39]
                    ]
                '''

        # 7.计算初始混合权重，每个簇占比作为初始pi
        pi = np.array([len(c) * 1.0 / len(data) for c in clusters], dtype=np.float64)
        return mu, sigma, pi

    def gaussian(self, x, mu, sigma):
        """
        （兼容保留，不是 EM/LLH 的主要实现）
        直接返回高斯密度值（非 log），数值上可能非常小。
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        sigma = np.asarray(sigma, dtype=np.float64)

        # .reshape(-1)
        # 作用：把数组变成一维（不管原来是什么形状）
        # -1 表示"自动计算这个维度的大小"

        D = x.shape[0]
        sigma = sigma + self.eps * np.eye(D)
        # np.eye(D) - 创建D维单位矩阵
        # self.eps * np.eye(D) - 创建很小的对角矩阵

        # 用 slogdet 更稳定
        sign, logdet = np.linalg.slogdet(sigma)
        # linalg 是 "linear algebra"（线性代数）的缩写
        # np.linalg 包含了各种线性代数运算的函数
        #np.linalg.slogdet()的返回值：
        # sign：行列式的符号（+1或-1）
        # logdet：行列式绝对值的自然对数

        # 检查sign
        if sign <= 0:  # 如果行列式为0或负数
            # 再加10倍的正则项
            sigma = sigma + 10 * self.eps * np.eye(D)
            # 重新计算
            sign, logdet = np.linalg.slogdet(sigma)

        inv_sigma = np.linalg.inv(sigma) #计算协方差矩阵的逆
        diff = x - mu  #计算与均值的差值
        mahal = diff.T @ inv_sigma @ diff  #计算马氏距离，@是矩阵乘法运算符

        logp = -0.5 * (D * np.log(2 * np.pi) + logdet + mahal)
        return float(np.exp(logp))
    #返回值是一个概率密度值，表示在给定高斯分布（均值 μ，协方差 Σ）下，观察到点 x 的概率密度

    def _log_gaussian(self, X, k):
        """
        计算第 k 个高斯分量对一批样本 X 的 log pdf:
          log N(X | mu_k, sigma_k)

        参数:
          X: (N, D)
          k: 分量编号

        返回:
          (N,) 每个样本的 log pdf
        """
        X = np.asarray(X, dtype=np.float64)
        mu = self.mu[k]  # (D,)

        # 协方差正则
        sigma = self._regularize_sigma(self.sigma[k])  # (D,D)

        # log|Sigma| 用 slogdet 稳定计算
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            sigma = sigma + 10 * self.eps * np.eye(self.dim)
            sign, logdet = np.linalg.slogdet(sigma)

        inv = np.linalg.inv(sigma)

        diff = X - mu  # (N, D)
        # 每行马氏距离: (x-mu)^T inv (x-mu)
        # 高效写法：sum((diff@inv)*diff, axis=1)
        mahal = np.sum((diff @ inv) * diff, axis=1)  # (N,)

        # log N = -0.5*(D log(2pi) + logdet + mahal)
        return -0.5 * (self.dim * np.log(2.0 * np.pi) + logdet + mahal)

    def calc_log_likelihood(self, X):
        """
        计算当前 GMM 对一批样本 X 的总对数似然：

          log p(X) = sum_n log( sum_k pi_k * N(x_n | mu_k, sigma_k) )

        参数:
          X: (N, D)

        返回:
          标量 log-likelihood
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Expected X shape (N,{self.dim}), got {X.shape}")

        # log(pi)
        log_pi = np.log(np.maximum(self.pi, 1e-300))

        N = X.shape[0]
        log_comp = np.empty((N, self.K), dtype=np.float64)

        # log_comp[n,k] = log(pi_k) + log N(x_n | ...)
        for k in range(self.K):
            log_comp[:, k] = log_pi[k] + self._log_gaussian(X, k)

        # log p(x_n) = logsumexp_k log_comp[n,k]
        log_px = _logsumexp(log_comp, axis=1)  # (N,)
        log_llh = float(np.sum(log_px))
        return log_llh

    def em_estimator(self, X):
        """
        对当前 GMM 做一次 EM 更新（一次迭代）：

        E-step:
          计算责任度 r[n,k]

        M-step:
          用 r[n,k] 更新 pi/mu/sigma

        参数:
          X: (N, D)

        返回:
          更新后的 log-likelihood（用于观察收敛，不影响主流程）
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Expected X shape (N,{self.dim}), got {X.shape}")

        N = X.shape[0]
        if N == 0:
            return -np.inf

        # ----------------------
        # E-step: responsibilities
        # ----------------------
        log_pi = np.log(np.maximum(self.pi, 1e-300))  # (K,)
        log_rho = np.empty((N, self.K), dtype=np.float64)

        # log_rho[n,k] = log(pi_k) + log N_k(x_n)
        for k in range(self.K):
            log_rho[:, k] = log_pi[k] + self._log_gaussian(X, k)

        # log_norm[n] = log sum_k exp(log_rho[n,k])
        log_norm = _logsumexp(log_rho, axis=1, keepdims=True)  # (N,1)

        # log_resp = log_rho - log_norm
        # resp = exp(log_resp)
        log_resp = log_rho - log_norm
        resp = np.exp(log_resp)  # (N,K)

        # ----------------------
        # M-step: update parameters
        # ----------------------
        Nk = np.sum(resp, axis=0)         # (K,)
        Nk = np.maximum(Nk, 1e-12)        # 防止除 0

        # 更新 pi
        self.pi = Nk / np.sum(Nk)
        self._normalize_pi()

        # 更新 mu: (K,D) = (K,N)@(N,D) / Nk
        self.mu = (resp.T @ X) / Nk[:, None]

        # 更新 sigma（全协方差）
        new_sigma = np.empty((self.K, self.dim, self.dim), dtype=np.float64)
        for k in range(self.K):
            diff = X - self.mu[k]               # (N,D)
            wdiff = diff * resp[:, k:k+1]       # (N,D) 每行乘以 r[n,k]
            cov = (wdiff.T @ diff) / Nk[k]      # (D,D)
            new_sigma[k] = self._regularize_sigma(cov)

        self.sigma = new_sigma
        self._regularize_all_sigmas()

        # 返回更新后的 log-likelihood
        log_llh = self.calc_log_likelihood(X)
        return log_llh


def train(gmms, num_iterations=num_iterations):
    """
    训练阶段：
      - 读取训练 feats/text
      - 对每个 target：
          取出该 target 的所有帧 feats (N,39)
          运行 num_iterations 次 EM
    """
    dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text.txt')

    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)
        for _ in range(num_iterations):
            _ = gmms[target].em_estimator(feats)
    return gmms


def test(gmms):
    """
    测试阶段：
      - 读取测试 feats/text
      - 对每条 utterance:
          计算 11 个模型的 log-likelihood
          取最大者为预测
      - 统计 accuracy
    """
    correction_num = 0
    error_num = 0

    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text.txt')

    # 构建 utt -> true target
    dict_utt2target = {}
    for target in targets:
        for utt in dict_target2utt[target]:
            dict_utt2target[utt] = target

    # 对每条 utt 进行分类
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])  # (T,39)
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))

        predict_target = targets[scores.index(max(scores))]

        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1

    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


def main():
    # 初始化 11 个类别的 GMM
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, K=num_gaussian)

    # 训练
    gmms = train(gmms)

    # 测试
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)

    # 输出 acc.txt
    with open('acc.txt', 'w') as fid:
        fid.write(str(acc))


if __name__ == '__main__':
    main()