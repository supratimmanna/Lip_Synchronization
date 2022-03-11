N = int(input())
for i in range(N):
    lst = list(map(int, input().split(' ')))
    ns = lst[0]
    avg_marks = 0
    for m in lst[1:]:
        avg_marks = avg_marks+m
        
    avg_marks = avg_marks/len(lst[1:])
    gt = sum(i > avg_marks for i in lst[1:])
    print(format(gt/len(lst[1:])*100, ".3f")+str('%'))