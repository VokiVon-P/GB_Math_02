#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Lessons'))
	print(os.getcwd())
except:
	pass
#%%
import numpy as np

#%% [markdown]
# Даны значения величины заработной платы заемщиков банка (zp) 
# и значения их поведенческого кредитного скоринга (ks): 
# zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110], 
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]. 
# Используя математические операции, посчитать коэффициенты линейной регрессии, 
# приняв за X заработную плату (то есть, zp - признак), 
# а за y - значения скорингового балла (то есть, ks - целевая переменная). 
# Произвести расчет как с использованием intercept, так и без.


#%%
# количество наблюдений


zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110] )
ks  = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])
x = zp
y = ks
n = x.size
n

#%% [markdown]
# Для нахождения коэффициентов регрессии $a$ и $b$ воспользуемся приведенными выше формулами:

#%%
b = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x**2) - np.mean(x) ** 2)
b


#%%
def estimate(x, y):
    y_av = y.mean() 
    x_av = x.mean()
    
    b1 = ((x - x_av) * (y - y_av )).sum() / ((x - x_av) ** 2).sum()
    b0 = y_av - b1 * x_av 
    
    return b0, b1

def predict_estimated(x, b0, b1):
    
    return b0 + x * b1


#%% 
x
#%% 
y
#%% 
b0, b1 = estimate(x, y)
print(f"Коэффициет b0 = {b0} и b1 = {b1} ")
#%% 
plt.scatter(x, y)
plt.title("Regression")
plt.xlabel("x (feature)")
plt.ylabel("y (target)")
plt.plot(x, predict_estimated(x, b0, b1), color='red', label='prediction estimated')
plt.legend()
# Итак, уравнение регрессии имеет вид (коэффициенты округлены до сотых):

#%% [markdown]
# $$y = 77.9 + 0.87 \cdot x$$
#%% [markdown]
# С увеличением площади квартиры на 1 квадратный метр цена возрастет на 0.87 тысячи долларов.
#%% [markdown]
# Найдем коэффициент корреляции $r$ с помощью коэффициента $b$ и средних квадратического отклонения, посчитанного для массивов $x$ и $y$:

#%%
r = b * np.std(x) / np.std(y)
r

#%% [markdown]
# Найдем коэффициент детерминации $R^2$:

#%%
R2 = r**2
R2

#%% [markdown]
# Это означает, что 67.5% вариации цены на квартиру ($y$) объясняется вариацией фактора $x$ — площади квартиры.
#%% [markdown]
# С помощью этого уравнения регрессии посчитаем значения, предсказанные моделью значения цен на квартиры:

#%%
y_pred = a + b * x
y_pred

#%% [markdown]
# Качество модели найдем с помощью средней ошибки аппроксимации $\overline {A}$:

#%%
A_mean = 100 * np.mean(np.abs((y - y_pred) / y))
A_mean

#%% [markdown]
# Так как $\overline {A}$ равна 4%, что не превышает 8-10 %, модель хорошо описывает эмпирические данные. Для оценки значимости 
# 
# уравнения регрессии воспользуемся F-критерием Фишера. Найдем фактическое значение $F$-критерия ($F_{факт}$):

#%%
F_fact = (r**2 * (n - 2)) / (1 - r**2)
F_fact

#%% [markdown]
# При 5 % уровне значимости и степенях свободы $k_1 = 1$ и $k_2 = 12 - 2 = 10$ табличное значение критерия: $F_{кр} = 4.96$.
# 
# Так как $F_{факт} = 20.79 > F_{кр} = 4.96$, уравнение регрессии статистически значимо.
#%% [markdown]
# Для оценки статистической значимости параметров регрессии воспользуемся $t$-статистикой Стьюдента и также рассчитаем 
# 
# доверительные интервалы каждого из показателей. При $df = n - 2 = 12 - 2 = 10$ и $\alpha = 0.05$ получим
# 
# (см. <a href='https://statpsy.ru/t-student/t-test-tablica/'>Таблицу критических значений t-критерия Стьюдента</a>):
#%% [markdown]
# $$t_{кр} = 2.228$$
#%% [markdown]
# Определим стандартную ошибку $S_{ост}$ (переменная **s_residual**) и случайные ошибки $m_a, \; m_b$:

#%%
s_residual = np.sqrt(np.sum((y - y_pred)**2) / (n - 2))
m_a = s_residual * np.sqrt(np.sum(x ** 2)) / (n * np.std(x))
m_b = s_residual / (np.std(x) * np.sqrt(n))

print('s_residual = {}\nm_a = {}\nm_b = {}'.format(s_residual, m_a, m_b))

#%% [markdown]
# Вычислим наблюдаемые значения критерия $t_a$ и $t_b$:

#%%
t_a = a / m_a
t_a


#%%
t_b = b / m_b
t_b

#%% [markdown]
# Фактические значения t-статистики больше табличного значения:
#%% [markdown]
# $$t_a = 4.44 > t_{кр} = 2.28, \; t_b = 4.56 > t_{кр} = 2.28,$$
#%% [markdown]
# поэтому параметры $a$ и $b$ не случайно отличаются от нуля, то есть они статистически значимы.
#%% [markdown]
# Рассчитаем доверительные интервалы для параметров регрессии $a$ и $b$. Для этого определим предельную ошибку для каждого показателя ($\Delta_a$ и $\Delta_b$),
# 
# используя значение $t_{кр}$, равное 2.28 (переменная **t_cr**):

#%%
t_cr = 2.28


#%%
delta_a = t_cr * m_a
delta_a


#%%
delta_b = t_cr * m_b
delta_b

#%% [markdown]
# Найдем границы доверительных интервалов $\gamma_{a_{min}}, \gamma_{a_{max}}, \gamma_{b_{min}}, \gamma_{b_{max}}$:

#%%
gamma_a_min = a - delta_a
gamma_a_min


#%%
gamma_a_max = a + delta_a
gamma_a_max


#%%
gamma_b_min = b - delta_b
gamma_b_min


#%%
gamma_b_max = b + delta_b
gamma_b_max

#%% [markdown]
# Приходим к выводу о том, что с вероятностью $p = 1 - \alpha = 0.95$ параметры $a$ и $b$, находясь в указанных границах, 
# 
# являются статистически значимыми и отличны от нуля. Поместим исходные и предсказанные данные в датафрейм **df**:

#%%
import pandas as pd

df = pd.DataFrame({'x': x, 'y': y, 'y_pred': y_pred}, columns=['x', 'y', 'y_pred'])
df

#%% [markdown]
# Отсортируем значения по полю **x**:

#%%
df = df.sort_values('x')
df

#%% [markdown]
# Построим на одном графике исходные данные и теоретическую прямую, построенную по уравнению регрессии:

#%%
import matplotlib.pyplot as plt

plt.scatter(df['x'], df['y'])
plt.plot(df['x'], df['y_pred'])
plt.xlabel('Площадь квартиры (кв. м.)')
plt.ylabel('Цена квартиры (тыс. долларов.)')
plt.show()


#%%



