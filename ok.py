troom, tcond = input().split(" ")
mode = input()
troom = int(troom)
tcond = int(tcond)

if mode == 'heat':
    if tcond > troom:
        print(tcond)
    else:
        print(troom)
elif mode == 'freeze':
    if troom > tcond:
        print(tcond)
    else:
        print(troom)
elif mode == 'auto':
    print(tcond)
elif mode == 'fan':
    print(troom)

