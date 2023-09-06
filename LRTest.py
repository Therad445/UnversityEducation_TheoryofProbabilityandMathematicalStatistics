# Обнаружение статистически значимых отличий в уровнях экспрессии генов больных раком

# Это задание поможет вам лучше разобраться в методах множественной проверки гипотез и позволит применить
# ваши знания на данных из реального биологического исследования.

# В этом задании вы:
#    вспомните, что такое t-критерий Стьюдента и для чего он применяется
#    сможете применить технику множественной проверки гипотез и увидеть собственными глазами, как она работает
#       на реальных данных
#    почувствуете разницу в результатах применения различных методов поправки на множественную проверку

# Основные библиотеки и используемые методы:
#   Библиотека scipy и основные статистические функции:
#       http://docs.scipy.org/doc/scipy/reference/stats.html#statistical-functions
#   Библиотека statmodels для методов коррекции при множественном сравнении:
#       http://statsmodels.sourceforge.net/devel/stats.html
# Статья, в которой рассматриваются примеры использования statsmodels для множественной проверки гипотез:
#       http://jpktd.blogspot.ru/2013/04/multiple-testing-p-value-corrections-in.html

# Описание используемых данных
# Данные для этой задачи взяты из исследования, проведенного в Stanford School of Medicine.
# В исследовании была предпринята попытка выявить набор генов, которые позволили бы более точно
# диагностировать возникновение рака груди на самых ранних стадиях.
# В эксперименте принимали участие 24 человек, у которых не было рака груди (normal), 25 человек, у которых это
# заболевание было диагностировано на ранней стадии (early neoplasia), и 23 человека с сильно выраженными
# симптомами (cancer).
# Ученые провели секвенирование биологического материала испытуемых, чтобы понять, какие из этих генов наиболее
# активны в клетках больных людей.
# Секвенирование — это определение степени активности генов в анализируемом образце с помощью подсчёта
# количества соответствующей каждому гену РНК.
# В данных для этого задания вы найдете именно эту количественную меру активности каждого из 15748 генов у
# каждого из 72 человек, принимавших участие в эксперименте.
# Вам нужно будет определить те гены, активность которых у людей в разных стадиях заболевания отличается
# статистически значимо.
# Кроме того, вам нужно будет оценить не только статистическую, но и практическую значимость этих результатов,
# которая часто используется в подобных исследованиях.
# Диагноз человека содержится в столбце под названием "Diagnosis".

# Практическая значимость изменения
# Цель исследований — найти гены, средняя экспрессия которых отличается не только статистически значимо,
# но и достаточно сильно. В экспрессионных исследованиях для этого часто используется метрика, которая
# называется fold change (кратность изменения). Определяется она следующим образом:
#    Fc(C,T)=T/C,T>C; −C/T,T<C
#    где C,T — средние значения экспрессии гена в control и treatment группах соответственно.
# По сути, fold change показывает, во сколько раз отличаются средние двух выборок.

#%%
def write_answer(file_name, answer):
    with open(file_name, "w") as fout: #..\..\Results\
        fout.write(str(answer))
#%%
import pandas as pd
import numpy as np
import scipy
from statsmodels.stats.weightstats import *
frame = pd.read_csv("gene_high_throughput_sequencing.csv", sep=",", header=0)
frame.head()


# Инструкции к решению задачи
# Задание состоит из трёх частей. Если не сказано обратное, то уровень значимости нужно принять равным 0.05.


# Часть 1: применение t-критерия Стьюдента
# В первой части вам нужно будет применить критерий Стьюдента для проверки гипотезы о равенстве средних в двух
# независимых выборках. Применить критерий для каждого гена нужно будет дважды:
#   для групп normal (control) и early neoplasia (treatment)
#   для групп early neoplasia (control) и cancer (treatment)
# В качестве ответа в этой части задания необходимо указать количество статистически значимых отличий, которые
# вы нашли с помощью t-критерия Стьюдента, то есть число генов, у которых p-value этого теста оказался меньше,
# чем уровень значимости.
#%%
print "Diagnosis values: "
print set(frame["Diagnosis"])
#%%
normal_neoplasia = frame[frame["Diagnosis"] == "normal"].drop(["Patient_id", "Diagnosis"], axis=1)
early_neoplasia = frame[frame["Diagnosis"] == "early neoplasia"].drop(["Patient_id", "Diagnosis"], axis=1)
cancer = frame[frame["Diagnosis"] == "cancer"].drop(["Patient_id", "Diagnosis"], axis=1)
data_columns = frame.columns.drop(["Patient_id", "Diagnosis"])
print "Normal neoplasia count: %i\tRow size: %i" % normal_neoplasia.shape
print "Early neoplasia count: %i\tRow size: %i" % early_neoplasia.shape
print "Cancer count: %i\tRow size: %i" % cancer.shape
# Для того, чтобы использовать двухвыборочный критерий Стьюдента, убедимся, что распределения в выборках
# существенно не отличаются от нормальных.
# Критерий Шапиро-Уилка:
# H0: среднее значение РНК в генах распределено нормально
# H1: не нормально.
#%%
print "Shapiro-Wilk normality test, W-statistic:"
for col_name in data_columns:
    print "\tNormal neoplasia \"" + col_name + "\": %f, p-value: %f" % stats.shapiro(normal_neoplasia[col_name])
    print "\tEarly neoplasia \"" + col_name + "\": %f, p-value: %f" % stats.shapiro(early_neoplasia[col_name])
    print "\tCancer  \"" + col_name + "\": %f, p-value: %f" % stats.shapiro(cancer[col_name])
# Не все значения распределены нормально

# Критерий Стьюдента:
# H0: средние значения РНК в генах распределено одинаково.
# H1: не одинаково.
#%%
def compare_genes_diffs(right, left, columns):
    return [scipy.stats.ttest_ind(right[col_name], left[col_name], equal_var = False).pvalue for col_name in columns]
#%%
normal_early_diff_genes = compare_genes_diffs(normal_neoplasia, early_neoplasia, data_columns)
normal_early_diff_genes_count = len(filter(lambda pvalue: pvalue < 0.05, normal_early_diff_genes))
print "T-Student test normal vs early neoplasia different distributed genes: %i" % normal_early_diff_genes_count
write_answer("normal_early.txt", normal_early_diff_genes_count)
#%%
early_cancer_diff_genes = compare_genes_diffs(early_neoplasia, cancer, data_columns)
early_cancer_diff_genes_count = len(filter(lambda pvalue: pvalue < 0.05, early_cancer_diff_genes))
print "T-Student test early neoplasia vs cancer different distributed genes: %i" % early_cancer_diff_genes_count
write_answer("early_cancer.txt", early_cancer_diff_genes_count)