n = int(input())
a = [int(input()) for _ in range(n)]
d = int(input())
x = int(input())

date_is_even = (d%2 == 0)

vio_count = 0
for digit in a:
    if date_is_even:
        if digit % 2 != 0:
            vio_count += 1 
    else:
        if digit % 2 == 0:
            vio_count += 1 
            
total_fine = vio_count * x 
print(total_fine if total_fine > 0 else 0)