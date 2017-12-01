from subprocess import PIPE,Popen
print('IN KEYPRESS')
sequence = b'''keydown Control_L
keydown Shift_L
key equal
keyup Control_L
keyup Shift_L
'''
for i in range(8):
    p = Popen(['xte'], stdin=PIPE)
    p.communicate(input=sequence)
