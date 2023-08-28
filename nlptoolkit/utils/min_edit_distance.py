from typing import List


def minEditDistance(source: str, target: str) -> int:
    """
    计算两个字符串的最小编辑距离

    参数:
      source: 源字符串
      target: 目标字符串
    
    返回:
      两个字符串的最小编辑距离,即源字符串转换成目标字符串的最少编辑操作次数
    """

    n = len(source)
    m = len(target)

    # 初始化编辑距离矩阵
    dist_matrix: List[List[int]] = [[0 for _ in range(m + 1)]
                                    for _ in range(n + 1)]

    for i in range(1, n + 1):
        dist_matrix[i][0] = i

    for j in range(1, m + 1):
        dist_matrix[0][j] = j

    # 动态规划计算最小编辑距离
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == target[j - 1]:
                dist_matrix[i][j] = dist_matrix[i - 1][j - 1]
            else:
                dist_matrix[i][j] = min(dist_matrix[i - 1][j] + 1,
                                        dist_matrix[i][j - 1] + 1,
                                        dist_matrix[i - 1][j - 1] + 1)

    return dist_matrix[n][m]


if __name__ == '__main__':
    import numpy as np
    a = 'intention'
    b = 'execution'
    d = minEditDistance(a, b)
    d = np.array(d)
    print(d)
