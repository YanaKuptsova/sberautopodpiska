# sberautopodpiska
Анализ данных сайта сервиса долгосрочной аренды автомобилей для физлиц. Предсказание совершения целевого действия.
Задача - предсказать совершение целевого действия
(ориентировочное значение ROC-AUC ~ 0.65) — факт совершения
пользователем целевого действия.
Упаковать получившуюся модель в сервис, который будет брать на
вход все атрибуты, типа utm_*, device_*, geo_*, и отдавать на выход
0/1 (1 — если пользователь совершит любое целевое действие).
