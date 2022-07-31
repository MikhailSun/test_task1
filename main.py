import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import makecharts as mc
import resistances as rs
#Исходные данные (все единицы в СИ, если не указано иное)
#Резервуар 1 (источник)
Dr1=3.
Hr1=10.
Sr1=np.pi*(Dr1/2)**2
Vr1=Sr1*Hr1
Pr1=100000. #Па, это только атмосферное давление без учета давления от столба жидкости
#Резервуар 2 (слив)
Dr2=2.
Hr2=2.
Sr2=np.pi*(Dr2/2)**2
Vr2=Sr2*Hr2
Pr2=100000. #Па
#труба 1 (от резервуара до насоса)
Lp1=1. #имеется ввиду в том числе перепад трубы по высоте, для простоты примем, что он равен физической длине трубы
Dp1=0.25
Fp1=np.pi*(Dp1/2)**2
dHf1=rs.dH_pipe #потери трения
#труба 2 (от клапана до слива)
Lp2=5. #имеется ввиду в том числе перепад трубы по высоте, для простоты примем, что он равен физической длине трубы
Dp2=0.3
Fp2=np.pi*(Dp2/2)**2
dHf2=rs.dH_pipe #потери трения
#насос (между трубой 1 и трубой 2)
Q = [0., 0.1, 0.2, 0.3, 0.4, 0.5] #объемный расход для характеристики насоса (м3/с)
H = [30., 29., 28., 25.,15., 1.] #напор для характеристики насоса, м
Pump_H_Q = interp1d(Q, H,bounds_error=False,fill_value=(15,0),kind='quadratic') #функция с характеристикой насоса, по умолчанию они использует линейню интерполяцию
#клапан
dhValve=rs.dH_valve2

dP_cavitation=30000
Rho=1000. #плотность жидкости, кг/м3 (принимаем жидкость несжимаемой)
Nu=1.006e-6 #вязкость жидкости, м2/с
g=9.81
dt=1#шаг расчета
T=100 #общее время расчета
time_pump= {5:True, #время включения/выключения насоса
            90:False}
time_valve= {10:1, #время включения/выключения насоса,
            25:0.75,
             50:0.5,
             60:0.2,
             70:0.1,
             80:0.0}

#Инициализация первоначальных условий:
#делаем допущение, что труба изначально полностью заполнена жидкостью, жидкость покоится на месте
#давление в обоих резервуарах равно атмосферному (параметры Pr1 и Pr2)
pump_status=False #насос выключен
valve_status=1 #клапан открыт
Q_init=0. #объемный расход
_Vr2=0 #объем жидкости во 2м резервуаре

#Массивы для записи результатов:
P_pump_inlet=[np.nan]
P_pump_outlet=[np.nan]
P4_array=[Pr2/100000]
Q=[Q_init]
V_pipe1_inlet=[0]
V_pipe1_outlet=[0]
V_pipe2_inlet=[0]
V_pipe2_outlet=[0]
time=[0]
dH_valve_result=[0]
dH_pipe1=[0]
dH_pipe2=[0]
E2_array=[np.nan]
E3_array=[np.nan]
Ep_array=[np.nan]
Hp_array=[np.nan]
Hr1_array=[Hr1]
Hr2_array=[0]

#подфункция для расчета всех основных параметров в трубопроводе. Испоьлзуется для итерационного поиска этих самых параметров.
#необходимость использования итерационного поиска вызвана тем, что рассчет основывается на величине расход жидкости, которая изначально неизвестна
#на выходе эта функция выдает невязку по давлению на выходе из трубы и давлению в точке, куда выходит эта труба
def iteration(Q_,pump_status,valve_status,dt,Hr1):
    #находим напор насоса
    if pump_status:
        Hp=Pump_H_Q(Q_)
    else:
        Hp=0.
    #уравнение Бернулли для трубы 1 p1/Rho/g + h_reservoir == Lp1 + p2/Rho/g + V2**2/2/g  + dHf1 (+ a2*Lp1/g - нестационарный член)
    V2=Q_/Fp1 #находим в первом приближении скорость исходя из значения на предыдущем шаге
    dHfriction_pipe1 = dHf1(V2, Dp1, Nu, Lp1, g) #потери давления на трение в трубе 1
    P2 = (Pr1 / Rho / g + Hr1 - Lp1 - V2 ** 2 / 2 / g - dHfriction_pipe1) * Rho * g
    #найдем уд.энергию потока перед насосом
    H2=P2/Rho/g + V2**2/2/g
    E2 = H2*g #дж/кг
    # уд. энергия подводимая насосом за интервал времени dt (из уравнения мощности насоса N = Q*Rho*g*H или E/dt = Volume/dt*Rho*g*H)
    Ep=(Hp*g)
    # найдем уд.энергию потока за насосом
    # H3=H2+Hp
    E3 = E2 + Ep
    #параметры потока во второй трубе
    V3 = Q_ / Fp2
    V4 = Q_ / Fp2
    # dHvalve = dhValve(V3,g,Dp2,Dp2,valve_status)
    dHvalve = dhValve(V3,Dp2,Nu,g,valve_status)
    dHfriction_pipe2 = dHf2(V4, Dp2, Nu, Lp2, g)
    # P4 = (H3 - (Lp2 + V4 ** 2 / 2 / g + dHfriction_pipe2 + dHvalve)) * Rho * g
    # P4=(E3-V4**2/2-g*Lp2-dHfriction_pipe2-dHvalve)*Rho
    #из уравнения Бернули для трубы 2: H2+H_pump == p3/Rho/g + V3**2/2/g + dH_valve == Lp2 + p4/Rho/g + V4**2/2/g  + dHf2 + dH_valve
    P4=((H2+Hp)-(Lp2 + V4**2/2/g  + dHfriction_pipe2 + dHvalve))*Rho*g


    # P4 = (E3 - g * Lp2 - dHfriction_pipe2 - dHvalve) * Rho
    error=P4-Pr2 #невязка по давлению на выходе из трубопровода, она д.б. равна 0, для этого нужно найти правильное значение расхода
    return error

#Расчет
for i in range(1,int(T/dt)):
    t=i*dt
    time.append(t)
    if i%1000==0:
        print(i)
    #проверяем статус насоса
    for t_,pump_status_ in time_pump.items():
        if (abs(t_-t)<dt*0.5):
            pump_status=pump_status_
    #проверяем статус клапана
    for t_,valve_status_ in time_valve.items():
        if (abs(t_-t)<dt*0.5):
            valve_status=valve_status_
    #принимаем, что если насос выключен, то расход равен 0 (что-то вроде обратного клапана)
    if pump_status:
        #итерационно находим расход жидкости (подробнее написано в комментарии к функции iteration), в качестве первого приближения используетяс расход на предыдущем временном шаге Q[i - 1]
        #для поиска используется стандартная библиотека scipy.optimize.root_scalar метод секущих - дискуссионный остается вопрос, какой метод лучше использовать для конкретной задачи
        if Q[i - 1]==0:
            _x0=0.00001
            _x1=0.00002
            # Qnew = root_scalar(iteration, x0=_x0, x1=_x1, method='secant', args=(pump_status, valve_status))
        else:
            _x0=Q[i - 1]
            _x1=Q[i - 1]*1.0001
        #тут я пробовал разные методы поиска корня, т.к. были проблемы с поиском расхода при изменении положения клапана
        Qnew=root_scalar(iteration,x0=_x0,x1=_x1,method='secant',args=(pump_status,valve_status,dt,Hr1))
        # Qnew = root_scalar(iteration, bracket=(0, 10*Q[i - 1]), method='toms748', args=(pump_status, valve_status,dt))
        # if Qnew.root<0:
        #     Qnew = root_scalar(iteration, bracket=(0,Q[i - 1]), method='toms748', args=(pump_status, valve_status))
        #     Qnew = root_scalar(iteration, x0=0.000001, x1=0.000002, method='secant', args=(pump_status, valve_status,dt))
        Q.append(Qnew.root)
    else:
        # Qnew = root_scalar(iteration, x0=0.0001, x1=0.0002, method='secant', args=(pump_status, valve_status, dt))
        Q.append(0.)
    print(f'Q={Q[i]} time={t}')
    # находим напор насоса
    if pump_status:
        Hp=Pump_H_Q(Q[i])
    else:
        Hp=0.
    # уравнение Бернулли для трубы 1 p1/Rho/g == Lp1 + p2/Rho/g + V2**2/2/g + a2*Lp1/g + dHf1
    V2 = Q[i] / Fp1
    dHfriction_pipe1=dHf1(V2,Dp1,Nu,Lp1,g)
    P2 = (Pr1 / Rho / g + Hr1 - Lp1 - V2 ** 2 / 2 / g - dHfriction_pipe1) * Rho * g
    if dP_cavitation < Pr1 -P2 :
        print(f'Warning! Possible cavitation P2={P2}')
    # найдем энергию потока перед насосом и за ним
    H2 = P2 / Rho / g + V2 ** 2 / 2 / g
    E2 = H2 * g
    Ep = (Hp * g )
    E3 = E2 + Ep
    H3 = H2 + Hp
    # параметры потока во второй трубе, также на основе уравнения Бернулли H2+H_pump == p3/Rho/g + V3**2/2/g + dH_valve == Lp2 + p4/Rho/g + V4**2/2/g + a4*Lp1/g + dHf2 + dH_valve
    V3 = Q[i] / Fp2
    # dHvalve=dhValve(V3,g,Dp2,Dp2,valve_status)
    dHvalve = dhValve(V3, Dp2, Nu, g, valve_status)
    P3 = (H3 - (V3**2/2/g + dHvalve))*Rho*g
    V4 = Q[i] / Fp2
    dHfriction_pipe2 = dHf2(V4,Dp2,Nu,Lp2,g)
    # P4 = (H3 - (Lp2 + V4 ** 2 / 2 / g + dHfriction_pipe2 + dHvalve)) * Rho * g
    P4=(E3-V4**2/2-g*Lp2-dHfriction_pipe2-dHvalve)*Rho
    #считаем опустошение/заполнение резервуаров
    dVr=Q[i]*dt
    Vr1-=dVr
    _Vr2+=dVr
    #уровень жидкости в резервуарах
    Hr1=Vr1/Sr1
    Hr2 = _Vr2 / Sr2


    #сохраняем результаты
    P_pump_inlet.append(P2/100000)
    P_pump_outlet.append(P3/100000)
    P4_array.append(P4/100000)
    V_pipe1_outlet.append(V2)
    V_pipe2_inlet.append(V3)
    V_pipe2_outlet.append(V4)
    dH_valve_result.append(dHvalve)
    E2_array.append(E2)
    E3_array.append(E3)
    Ep_array.append(Ep)
    Hp_array.append(Hp)
    dH_pipe1.append(dHfriction_pipe1)
    dH_pipe2.append(dHfriction_pipe2)
    Hr1_array.append(Hr1)
    Hr2_array.append(Hr2)


fig1=mc.Chart(points_for_plot=[{'x':time,'y':P_pump_inlet,'label':'P_pump_inlet'},{'x':time,'y':P_pump_outlet,'label':'P_pump_outlet'}],xlabel='t',ylabel='P, bar',title='Pressures', dpi=150,figure_size=(5,5))
fig2=mc.Chart(points_for_plot=[{'x':time,'y':Q,'label':'Q'}],xlabel='t',ylabel='Q, m3/s',title='Volume flow', dpi=150,figure_size=(5,5))
fig3=mc.Chart(points_for_plot=[{'x':time,'y':V_pipe1_outlet,'label':'V_pipe1_outlet'},{'x':time,'y':V_pipe2_inlet,'label':'V_pipe2_inlet'},{'x':time,'y':V_pipe2_outlet,'label':'V_pipe2_outlet'}],xlabel='t',ylabel='V, m/s',title='Velocities', dpi=150,figure_size=(5,5))
fig4=mc.Chart(points_for_plot=[{'x':time,'y':dH_valve_result,'label':'dH_valve_result'}],xlabel='t',ylabel='dH_valve_result, m',title='Pressure drop in valve', dpi=150,figure_size=(5,5))
# fig5=mc.Chart(points_for_plot=[{'x':time,'y':E2_array,'label':'E2_array'},{'x':time,'y':E3_array,'label':'E3_array'},{'x':time,'y':Ep_array,'label':'Ep_array'}],xlabel='t',ylabel='E', dpi=150,figure_size=(5,5))
fig6=mc.Chart(points_for_plot=[{'x':time,'y':Hp_array,'label':'Hp_array'}],xlabel='t',ylabel='H, m',title='Pump head', dpi=150,figure_size=(5,5))
fig7=mc.Chart(points_for_plot=[{'x':time,'y':dH_pipe1,'label':'dH_pipe1'},{'x':time,'y':dH_pipe2,'label':'dH_pipe2'}],xlabel='t',ylabel='dH, m',title='Pressure drop in pipes', dpi=150,figure_size=(5,5))
fig8=mc.Chart(points_for_plot=[{'x':time,'y':Hr1_array,'label':'Hr1_array'},{'x':time,'y':Hr2_array,'label':'Hr2_array'}],xlabel='t',ylabel='Hr, m',title='Levels in reservoirs', dpi=150,figure_size=(5,5))

Q_array=np.linspace(0,0.5,100)
H_array=[Pump_H_Q(Q) for Q in Q_array]
fig9=mc.Chart(points_for_plot=[{'x':Q_array,'y':H_array,'label':'Pump map'}],xlabel='Q, m3/s',ylabel='H, m',title='Pump map', dpi=150,figure_size=(5,5))
plt.show()


