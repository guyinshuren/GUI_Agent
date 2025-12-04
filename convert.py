def compute_values(a, b):
    result1 = a / 1276 * 1000
    result2 = b / 2848 * 1000
    return result1, result2

# 示例：输入两个数
x = float(input("请输入第一个数："))
y = float(input("请输入第二个数："))

r1, r2 = compute_values(x, y)

print("第一个数结果：", r1)
print("第二个数结果：", r2)
