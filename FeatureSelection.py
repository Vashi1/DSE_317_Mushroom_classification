from sklearn.decomposition import PCA


def PCA_data(data_imp):
    pca = PCA(10)
    data_pca = pca.fit_transform(data_imp)
    return data_pca


from sklearn.decomposition import KernelPCA


def Kernal_PCA(data_imp, k=5):
    kpca = KernelPCA(k)
    data_kpca = kpca.fit_transform(data_imp)
    return data_kpca


# Mention value of number of components
from sklearn.decomposition import TruncatedSVD


def TrSVD(data_imp, k=5):
    trsvd = TruncatedSVD(k)
    data_trSVD = trsvd.fit_transform(data_imp)
    return data_trSVD
