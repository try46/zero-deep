class Man:

    def __init__(self, name):
        self.name = name
        print("Initialize")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye" + self.name + "!")


m = Man("David")
m.hello()
m.goodbye()
