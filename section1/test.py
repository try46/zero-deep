print(1+2)

print(1 -2)

print( 4 * 5)

print(7 / 5)

print(3 ** 2)

## type

print(type(10))

print(type(2.718))

print(type("hello"))

x = 10

print(x)

x = 100

print(x)

y  = 3.14

print(x * y)

print(type(x * y))


## リスト

a = [1,2,3,4,5]

print(a)

print(len(a))

print(a[0])

print(a[4])

a[4] = 99
print(a[4])

print(a)

print(a[0:2])

print(a[1:])

print(a[:3])

print(a[:-1])

print(a[:2])

## ディクショナリ

me = {"height":100}

print(me["height"])

me["weight"] = 70

print(me)

#Boolean

hungry = True

sleepy = False

print(type(hungry))

print(not hungry)

print(hungry and sleepy)

print(hungry or sleepy)

# if文

if hungry:
    print("I'm a hungry")

hungry = False

if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry")
    print("I'm a sleepy")


# for文

for i in [1,2,3]:
    print(i)

def hello():
    print("Hello World")

hello()

def hello1(object):
    print("Hello " + object+"!")

hello1("cat")



