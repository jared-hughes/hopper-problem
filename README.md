# Hopper Problem

The problem is : You have n people and give each a ball to put into a hopper, which randomly takes one out each time. Whenever someone is picked, their ball gets re-fed into the hopper, and everyone else doubles the amount of balls they have in the hopper.
Provided that they all start with 1 ball inside of the hopper, on average, how long will it take for each person to have an equal amount of balls in the hopper again?
(Answer in terms of n)

## Miscellaneous notes

Accurate N=2 using series expansion: 3.2832651213103072

Best values:

```
n k    → hopperN(n, k)
2 1000 → 3.2832651213103077
3 1000 → ≥9.012315262795905
4 1000 → ≥23.10316401473495
5 1000 → ≥57.120945265786446
6 2500 → ≥138.09967578103917
7 10000 → ≥328.83258522058964
8 15000 → ≥774.401520361932
9 20000 → ≥1808.5385603901827
10 20000 → ≥4196.0587932892395
11 20000 → ≥9658.46111659968
12 20000 → ≥18997.898543469342
```

Rate of convergence:

```
n\k                250                500               1000
  2    3.2832652100438    3.2832652100438    3.2832652100438
  3    9.0123166479381    9.0123166479381    9.0123166479381
  4   23.1031739020325   23.1031739020327   23.1031739020328
  5   57.1182431083783   57.1210048911313   57.1210051059042
  6  132.2015389033584  138.0232792325331  138.0999638617856
  7  111.8224744348625  291.9284826308397  328.0235836073838
```
