import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv('creditcard.csv')
mp = []
pm = []
i = 0  # берем для анализа отрезок по 10000 транзакций
j = 10000
lapse = 10000
my_serias = df['Class']
frod = df.loc[df['Class'] == 1]  # выбрали из файла только фродовые транзакции
nfrod = df.loc[df['Class'] == 0]  # выбрали из файла не фродовые транзакции


def func(x, y):
    k = 0
    for number in my_serias[x:y]:
        if number == 1:  # просматриваем фродовые транзакции
            k += 1
    mp.insert(0, k)


for num in range(27):  # судя из файла у нас 284000 транзакций, анализ пойдет по 280000 транзакций с сегментами по
    # 10000 транзакций
    func(i, j)
    i += lapse
    j += lapse
var = mp[::-1]
print("Всего транзакций у нас", len(df), "Из которых фродовыми являются", len(frod))
print("Мы провели анализ 280000 транзакций:")
print("Сегмент с наибольшей долей фродовых находится в промежутке между", var.index(max(var)) * lapse, "и",
      (var.index(max(var)) + 1) * lapse, "транзакциями")
print("Сегмент с наименьшей долей фродовых находится в промежутке между", var.index(min(var)) * lapse, "и",
      (var.index(min(var)) + 1) * lapse, "транзакциями")
# Извиняюсь за большой код, не нашел как сделать это все циклом Как вариант для нахождения значимых факторов
# предлагаю задейстовать скользящее среднее: Вычислить скользящее среднее для каждого фактора у фродовых и не
# фродовых транзакций и сравнить эти значения у каждого фактора и где разница ск.ср будет наибольшей те факторы
# назвать значимыми
frod = frod[
    ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
     'V18', 'V19',
     'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27']].rolling(2, win_type='triang').sum()
nfrod = nfrod[
    ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
     'V18', 'V19',
     'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27']].rolling(2, win_type='triang').sum()
pm.insert(0, math.fabs(math.fabs(frod['V1'].iloc[-1]) - math.fabs(nfrod['V1'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V2'].iloc[-1]) - math.fabs(nfrod['V2'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V3'].iloc[-1]) - math.fabs(nfrod['V3'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V4'].iloc[-1]) - math.fabs(nfrod['V4'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V5'].iloc[-1]) - math.fabs(nfrod['V5'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V6'].iloc[-1]) - math.fabs(nfrod['V6'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V7'].iloc[-1]) - math.fabs(nfrod['V7'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V8'].iloc[-1]) - math.fabs(nfrod['V8'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V9'].iloc[-1]) - math.fabs(nfrod['V9'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V10'].iloc[-1]) - math.fabs(nfrod['V10'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V11'].iloc[-1]) - math.fabs(nfrod['V11'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V12'].iloc[-1]) - math.fabs(nfrod['V12'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V13'].iloc[-1]) - math.fabs(nfrod['V13'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V14'].iloc[-1]) - math.fabs(nfrod['V14'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V15'].iloc[-1]) - math.fabs(nfrod['V15'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V16'].iloc[-1]) - math.fabs(nfrod['V16'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V17'].iloc[-1]) - math.fabs(nfrod['V17'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V18'].iloc[-1]) - math.fabs(nfrod['V18'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V19'].iloc[-1]) - math.fabs(nfrod['V19'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V20'].iloc[-1]) - math.fabs(nfrod['V20'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V21'].iloc[-1]) - math.fabs(nfrod['V21'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V22'].iloc[-1]) - math.fabs(nfrod['V22'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V23'].iloc[-1]) - math.fabs(nfrod['V23'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V24'].iloc[-1]) - math.fabs(nfrod['V24'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V25'].iloc[-1]) - math.fabs(nfrod['V25'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V26'].iloc[-1]) - math.fabs(nfrod['V26'].iloc[-1])))
pm.insert(0, math.fabs(math.fabs(frod['V27'].iloc[-1]) - math.fabs(nfrod['V27'].iloc[-1])))
# Найдем для примера 3 наиболее значимых факторов
print(pm)
f: int = 1
for number in range(3):
    print("Наиболее значимым фактором является фактор V", pm.index(max(pm)) + f, '')
    pm.pop(pm.index(max(pm)))
    f += 1
# я прочитал статью на хабре про методы отбора фич, но решил воспользоваться подсказкой про скользящее среднее,
# может быть я не так это применил =)
