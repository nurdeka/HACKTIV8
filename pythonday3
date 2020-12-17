#!/usr/bin/env python
# coding: utf-8

# In[25]:


# menghitung arround lingkaran
def CircleArroundFunction (jariJari):
    return 2*jariJari*22/7

# menghitung area lingkaran
def CircleAreaFunction (jariJari):
    return 22/7*(jariJari**2)

print('Menghitung Luas Lingkaran dengan Jari jari 7')
print(CircleAreaFunction(7))

print('Menghitung Keliling Lingkaran dengan Jari jari 28')
print(CircleArroundFunction(28))

def CylinderVolumeFunction(jariJari,tinggi):
    return CircleAreaFunction(jariJari) * tinggi

def BallVolumeFunction(jariJari):
    return 22/7*4/3*(jariJari**3)

print('Menghitung Volume Tabung dengan Tinggi 15 dan Jari jari 7')
print(CylinderVolumeFunction(7,15))

print('Menghitung Volume Bola Bumi')
print(BallVolumeFunction(20000))

class Circle():
    def __init__(self, r):
        self.radius = r

    def area(self):
        return self.radius**2*3.14
    
    def perimeter(self):
        return 2*self.radius*3.14

NewCircle = Circle(8)
print('Menghitung Luas Class Lingkaran:')
print(NewCircle.area())
print('Menghitung Keliling Class Lingkaran:')
print(NewCircle.perimeter())

class Ractangle():
    def __init__(self, s):
        self.sisi = s
    
    def area(self):
        return self.sisi**2
    
    def perimeter(self):
        return 4*self.sisi

NewRactangle = Ractangle(5)
print('Menghitung Luas Class Persegi :')
print(NewRactangle.area())
print('Menghitung Keliling Class Persegi :')
print(NewRactangle.perimeter())
    
    
class Triangle():
    def __init__(self, sisi1,sisi2,sisi3,angle1,angle2,angle3):
        self.a = sisi1
        self.b = sisi2
        self.c = sisi3
        self.a1= angle1 # sudut antara a dan b
        self.a2= angle2 # sudut antara b dan c
        self.a3= angle3 # sudut antara a dan c
        
    def area(self):
        if self.a1 == 90:
            if self.a2+self.a3 == 90:
                return 1/2*self.a*self.b
            else:
                print ('Bukan Segitiga')
                return -1
        elif self.a2 == 90:
            if a1+a3 == 90:
                return 1/2*self.b*self.c
            else:
                print ('Bukan Segitiga')
                return -1
        elif self.a3 == 90:
            if self.a1+self.a3 == 90:
                return 1/2*self.a*self.c
            else:
                print ('Bukan Segitiga')
                return -1
        else: 
            print ('Bukan Segitiga Siku Siku')
            if (self.a1 == self.a2 == self.a3) & (self.a == self.b == self.c):
                print ('Segitiga Sama Sisi')
                return self.a**2/4*1.732 # rumus luas = a^2/4*sqrt(3)
            else:
                print('Bukan Segitiga sama sisi')
                return -1
            
newTriangle = Triangle(3,4,5,90,30,60)  
print('Luas Segitiga :')
print(newTriangle.area())

newTriangle2 = Triangle(3,4,5,90,30,30)
print('Luas Segitiga :')
print(newTriangle2.area())

newTriangle3 = Triangle(5,5,5,60,60,60)
print('Luas Segitiga :')
print(newTriangle3.area())

newTriangle4 = Triangle(5,4,4,60,60,60)
print('Luas Segitiga :')
print(newTriangle4.area())


# In[14]:


# Create a class, the blueprint for your object
class MyTriangle(object):
	
    # The __init__ function is called automatically when an object is created
	def __init__(self, base=0, height=0): # 0 for b & h if nothing is passed in
		self.base = base
		self.height = height
	
	def setBase( self, base ):
		self.base = base
        
	def getBase( self ):
		return self.base
		
	def setHeight( self, height ):
		self.height = height
	
	def getHeight(self):
		return self.height
    
	def getArea(self):
		area = self.base * self.height / 2
		return area
		

        
# Create instances of your object (the "houses" made from your blueprint :)

# METHOD 1: Create and then set values
# Create new empty (0 base, 0 height) triangle object
t1 = MyTriangle()
# Use methods to set base and height 
t1.setBase(10)
t1.setHeight(5.234)
print (t1.getArea())

# METHOD 2: Pass base and height directly upon creating the object
t2 = MyTriangle(10,2.35)
print (t2.getArea())

# Using the getters
print (t2.getBase())
print (t2.getHeight())
print (t2.getArea())

# More useful example (the %5.4f means format as float value with up to 5 digits on the left and 4 after the decimal
print ("Triangle with base %5.4f and height %5.4f has area %5.4f" % ( t2.getBase(), t2.getHeight(), t2.getArea() ))


		


# In[ ]:


# Create a class, the blueprint for your object
class MyTriangle(object):
	
    # The __init__ function is called automatically when an object is created
	def __init__(self, base=0, height=0): # 0 for b & h if nothing is passed in
		self.base = base
		self.height = height
	
	def setBase( self, base ):
		self.base = base
        
	def getBase( self ):
		return self.base
		
	def setHeight( self, height ):
		self.height = height
	
	def getHeight(self):
		return self.height
    
	def getArea(self):
