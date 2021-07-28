# Hopper Problem

The problem is : You have n people and give each a ball to put into a hopper, which randomly takes one out each time. Whenever someone is picked, their ball gets re-fed into the hopper, and everyone else doubles the amount of balls they have in the hopper.
Provided that they all start with 1 ball inside of the hopper, on average, how long will it take for each person to have an equal amount of balls in the hopper again?
(Answer in terms of n)

## Miscellaneous notes

https://discord.com/channels/655972529030037504/663606178407907338/709595180294078527

https://discord.com/channels/655972529030037504/663606178407907338/869839530482606080

https://www.desmos.com/calculator/dcckwgmcn2

N=2: 3.2832651213103072

More monte carlo: https://www.desmos.com/calculator/vmekspgcat

Cyan Monte Carlo:

```
n  | avg draws
---|----------
0  |1
1  |1
2  |3.2794
3  |9.03678
4  |23.16816
5  |57.0008
```
