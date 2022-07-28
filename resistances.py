import numpy as np
from scipy.interpolate import interp1d
#сопротивление клапана. Я выбрал взял за основу дисковый клапан из Идельчика "Справочник по гидравлическим сопротивлениям".
#для тестового задания упростил - сопротивление задал с помщью трех кривых: 1я - состояние полностью прикрытого клапана, 2я - 50%, 3я - клапан открыт
#точки с графика снимал "на глаз", интерполяция везде линейная используется для простоты (но нет никакой сложности использовать более продвинутые аппроксимации)
curve1_Re=[1,100,1000,100000]
curve1_dzita=[1e4,2e3,9e2,9e2]
curve2_Re=[1,100,1000,100000]
curve2_dzita=[0.25e3,0.5e2,0.1e2,0.1e2]
curve3_Re=[1,100,1000,100000]
curve3_dzita=[0.5e2,5.,0.8,0.8]
curve1_f=interp1d(curve1_Re, curve1_dzita,bounds_error=False,fill_value=(1e4,9e2),kind='quadratic')
curve2_f=interp1d(curve2_Re, curve2_dzita,bounds_error=False,fill_value=(0.25e3,0.1e2),kind='quadratic')
curve3_f=interp1d(curve3_Re, curve3_dzita,bounds_error=False,fill_value=(0.5e2,0.8),kind='quadratic')
f_array=[curve1_f,curve2_f,curve3_f]

def dH_valve2(V,D,Nu,g,valve_status):
    Re = V * D / Nu
    Re = 1 if Re < 1 else Re
    x_array=[1.,0.5,0.]
    y_array=[f(Re) for f in f_array]
    dzita=interp1d(x_array, y_array,bounds_error=False) #коэффициент гидравлического сопротивления клапана
    return dzita(1-valve_status)*V**2/2/g

#потери в регулируемом клапане, заданном как поток через отверстие (Идельчик)
#valve_status = 0...1 (1 - клапан открыт, 0 - закрыт)
def dH_valve(V,g,Din,Dout,valve_status):
    F1=np.pi*(Din/2)**2
    F2 = np.pi * (Dout / 2) ** 2
    F0_=0.9*F1 #0.9 - произвольный коэффициент, т.е. проходное сечение клапана чуть меньше чем сечение входной трубы
    F0=F0_*valve_status
    dzita = (0.707 * (1 - F0 / F1) ** 0.375 + (1 - F0 / F2)) ** 2
    # if valve_status<0.2:
    #     dzita=(0.2-valve_status)/0.2*1000*dzita
    return dzita * V ** 2 / 2 / g

#коэффициент потерь трения по длине трубы
def Lam_function(Re):
    if Re<2300:
        return 64/Re #формула Пуазейля
    elif 2300<=Re<10000:
        return 0.316/(Re)**0.25 #ф.Блазиуса
    else:
        return 0.0032+0.221/Re**0.237 #ф.Никурадзе
    # return 64 / Re

#потери напора в трубе по ф.Дарси-Вейсбаха
def dH_pipe(V,D,Nu,L,g):
    Re=abs(V)*D/Nu
    Re = 1 if Re<1 else Re
    Lam=Lam_function(Re)
    return Lam*L*V**2/D/2/g
