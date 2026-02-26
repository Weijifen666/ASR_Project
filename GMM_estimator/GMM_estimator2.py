from utils import *
import numpy as np
import scipy.cluster.vq as vq


# =======================
# ADDED: 数值稳定的 log-sum-exp 工具函数
# =======================
def _logsumexp(a, axis=None, keepdims=False):
    """
    稳定计算 log(sum(exp(a)))，避免数值下溢/上溢
    """
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    # 处理全是 -inf 的情况
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    s = np.sum(np.exp(a - a_max), axis=axis, keepdims=True)
    out = a_max + np.log(s + 1e-300)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


num_gaussian = 5
num_iterations = 5
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


class GMM:
    def __init__(self, D, K=5):
        assert (D > 0)
        self.dim = D
        self.K = K
        # Kmeans Initial
        self.mu, self.sigma, self.pi = self.kmeans_initial()

    def kmeans_initial(self):
        mu = []
        sigma = []
        data = read_all_data('train/feats.scp')
        (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)
        clusters = [[] for i in range(self.K)]
        for (l, d) in zip(labels, data):
            clusters[l].append(d)

        for cluster in clusters:
            mu.append(np.mean(cluster, axis=0))
            sigma.append(np.cov(cluster, rowvar=0))
        pi = np.array([len(c) * 1.0 / len(data) for c in clusters])
        return mu, sigma, pi

    def gaussian(self, x, mu, sigma):
        """Calculate gaussion probability.

            :param x: The observed data, dim*1.
            :param mu: The mean vector of gaussian, dim*1
            :param sigma: The covariance matrix, dim*dim
            :return: the gaussion probability, scalor
        """
        D = x.shape[0]
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma + 0.0001)
        mahalanobis = np.dot(np.transpose(x - mu), inv_sigma)
        mahalanobis = np.dot(mahalanobis, (x - mu))
        const = 1 / ((2 * np.pi) ** (D / 2))
        return const * (det_sigma) ** (-0.5) * np.exp(-0.5 * mahalanobis)

    def calc_log_likelihood(self, X):
        """Calculate log likelihood of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of current model
        """

        log_llh = 0.0
        """
            FINISH by YOUSELF
        """
        # =======================
        # ADDED: 稳定计算 log-likelihood
        # =======================
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        eps = 1e-6

        # 将参数转成 numpy，便于矩阵运算
        mu = np.asarray(self.mu, dtype=np.float64)  # (K, D)
        sigma = np.asarray(self.sigma, dtype=np.float64)  # (K, D, D)
        pi = np.asarray(self.pi, dtype=np.float64)  # (K,)

        log_pi = np.log(np.maximum(pi, 1e-300))

        # 逐帧计算 log p(x_n)
        for n in range(N):
            log_comp = []
            for k in range(self.K):
                # 计算第 k 个高斯的 log pdf
                cov = sigma[k] + eps * np.eye(D)
                sign, logdet = np.linalg.slogdet(cov)
                inv_cov = np.linalg.inv(cov)
                diff = X[n] - mu[k]
                mahal = diff.T @ inv_cov @ diff
                log_gauss = -0.5 * (D * np.log(2 * np.pi) + logdet + mahal)

                log_comp.append(log_pi[k] + log_gauss)

            # log p(x_n) = logsumexp_k(log_comp)
            log_llh += _logsumexp(np.array(log_comp))

        return log_llh

    def em_estimator(self, X):
        """Update paramters of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of updated model
        """

        log_llh = 0.0
        """
            FINISH by YOUSELF
        """
        # =======================
        # ADDED: EM 更新（E-step + M-step）
        # =======================
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        eps = 1e-6

        mu = np.asarray(self.mu, dtype=np.float64)  # (K, D)
        sigma = np.asarray(self.sigma, dtype=np.float64)  # (K, D, D)
        pi = np.asarray(self.pi, dtype=np.float64)  # (K,)

        # ---------- E-step ----------
        log_rho = np.zeros((N, self.K), dtype=np.float64)
        for k in range(self.K):
            cov = sigma[k] + eps * np.eye(D)
            sign, logdet = np.linalg.slogdet(cov)
            inv_cov = np.linalg.inv(cov)

            diff = X - mu[k]  # (N, D)
            mahal = np.sum((diff @ inv_cov) * diff, axis=1)  # (N,)

            log_gauss = -0.5 * (D * np.log(2 * np.pi) + logdet + mahal)
            log_rho[:, k] = np.log(np.maximum(pi[k], 1e-300)) + log_gauss

        log_norm = _logsumexp(log_rho, axis=1, keepdims=True)  # (N,1)
        gamma = np.exp(log_rho - log_norm)  # (N,K)

        # ---------- M-step ----------
        Nk = np.sum(gamma, axis=0) + 1e-12  # (K,)
        pi_new = Nk / np.sum(Nk)

        mu_new = (gamma.T @ X) / Nk[:, None]  # (K,D)

        sigma_new = np.zeros((self.K, D, D), dtype=np.float64)
        for k in range(self.K):
            diff = X - mu_new[k]  # (N,D)
            wdiff = diff * gamma[:, k:k + 1]  # (N,D)
            cov = (wdiff.T @ diff) / Nk[k]  # (D,D)
            sigma_new[k] = cov + eps * np.eye(D)

        # 更新模型参数
        self.pi = pi_new
        self.mu = mu_new
        self.sigma = sigma_new

        log_llh = self.calc_log_likelihood(X)

        return log_llh


def train(gmms, num_iterations=num_iterations):
    dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')

    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)  #
        for i in range(num_iterations):
            log_llh = gmms[target].em_estimator(feats)
    return gmms


def test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
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
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, K=num_gaussian)  # Initial model
    gmms = train(gmms)
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()


if __name__ == '__main__':
    main()