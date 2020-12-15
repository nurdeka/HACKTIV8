#!/usr/bin/env python
# coding: utf-8

# In[1]:


kardus = ['aqua', 'sprite', 'cocacola', 'fanta']
print(kardus)


# In[2]:


botol = 'sprite'
print(botol)


# In[3]:


print(kardus[3])


# In[4]:


print(kardus[-1])


# In[5]:


kardus[0]='cleo'
print(kardus)


# In[6]:


kardus[-1]=80
print(kardus)


# In[7]:


print(kardus[1:3])


# In[8]:


print(kardus[1:4])


# In[9]:


kardus[1:3]=['chitato', 'chiki', 'cheetos', 'jetzet', 'qtela', 'oreo', 'malkist', 'doritos', 'taro']
print(kardus)


# In[10]:


kasir=('aqua', 'sprite', 'cocacola', 'fanta')
kasir[0]='chitato'


# In[11]:


kardus_label={'aqua':'nurdeka', 'sprite':'arif', 'cocacola':'thomas', 'fanta':'tio'}
print(kardus_label)


# In[12]:


print(kardus_label['aqua'])


# In[13]:


print(kardus_label['fanta'])


# In[14]:


kardus_label['dancow']='khafid'
print(kardus_label)


# In[15]:


kardus_label['baterai']=['ajie','husein']
print(kardus_label)


# In[16]:


kardus_label['mars']={'lampu': 'mars', 'gunting':'mars'}
print(kardus_label)


# In[21]:


if 'fanta' in kardus :
    print('fanta ada di dalam kardus')
else :
    print('fanta tidak ada di dalam kardus')
    print('kardus berisi ')
    print(kardus)


# In[77]:


while 'pringles' not in kardus:

    if 'pringles' in kardus :
        print('pringles ada di dalam kardus')
        kardus.remove('pringles')
        print('kardus berisi ')
        print(kardus)

    else :
        print('pringles tidak ada di dalam kardus')
        print('mencari taro')

        if 'taro' in kardus :
            print ('Taro ada di dalam kardus')
            kardus.remove('taro')

            if 'Qtela' in kardus :
                print ('Qtela ada di dalam kardus')
                kardus.remove('Qtela')

            elif 'oreo' in kardus :
                print ('Oreo ada di dalam kardus')
                kardus.remove('oreo')

            else :
                print ('Qtela dan Oreo tidak ada di dalam kardus')
                print ('membeli Qtela, oreo dan pringles')
                kardus.extend(['qtela','oreo','pringles'])
                print (' kardus berisi')
                print(kardus)

        else :
            print ('Taro tidak ada di dalam kardus')
            print ('membeli taro')
            kardus.append('taro')
            print('kardus berisi ')
            print(kardus)


# In[57]:


kardus.remove('pringles')


# In[80]:


import random 

a = 0

print('kardus berisi ' + str(len(kardus)) + ' barang')
print(kardus)
if 'pringles' in kardus:
    kardus.remove('pringles')
    print('mengambil pringles')

while 'pringles' not in kardus:
    a = a + 1
    if 'pringles' in kardus :
        print('pringles ada di dalam kardus')
        del kardus[random.randint(0,len(kardus)-1)]
        print('kardus berisi ' + str(len(kardus)) + ' barang')
        print(kardus)

    else :
        print('pringles tidak ada di dalam kardus')
        print('mencari taro')

        if 'taro' in kardus :
            print ('Taro ada di dalam kardus')
            del kardus[random.randint(0,len(kardus)-1)]

            if 'Qtela' in kardus :
                print ('Qtela ada di dalam kardus')
                del kardus[random.randint(0,len(kardus)-1)]
                print (kardus)

            elif 'oreo' in kardus :
                print ('Oreo ada di dalam kardus')
                del kardus[random.randint(0,len(kardus)-1)]
                print (kardus)

            else :
                print ('Qtela dan Oreo tidak ada di dalam kardus')
                print ('membeli Qtela, oreo dan pringles')
                kardus.extend(['qtela','oreo','pringles'])
                print('kardus berisi ' + str(len(kardus)) + ' barang')
                print(kardus)

        else :
            print ('Taro tidak ada di dalam kardus')
            print ('membeli taro')
            kardus.append('taro')
            print('kardus berisi ' + str(len(kardus)) + ' barang')
            print(kardus)
            
print('terjadi perulangan ' + str(a) + ' kali' )


# In[93]:


n = 210
while n > 0:
    n = n-40 # n -= 40
    print (n)


# In[94]:


while True:
    msg = input("Ketikan karakter:").lower()
    print(msg)
    if msg == "stop":
        break


# In[103]:


temp = input ("Ketikkan temperatur yang ingin dikonversi, eg.45F, 120C: ")
degree = int(temp[:1])
i_convertion = temp[-1]

if i_convertion == "C":
    result = int(round(9 * degree) / 5 + 32)
elif i_convertion == "F":
    result = int(round(degree - 32) * 5 / 9)
else:
    print("masukan input yang benar")
if i_convertion == "C":
    print("temperaturenya adalah ", result, "derajat Fahrenhait")
elif i_convertion == "F":
    print("temperaturenya adalah ", result, "derajat Celcius")


# In[ ]:
