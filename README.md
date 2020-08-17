# test-task
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naburov/test-task/blob/master/%D0%A4%D0%BE%D1%80%D0%BF%D0%BE%D1%81%D1%82%2C_%D1%82%D0%B5%D1%81%D1%82%D0%BE%D0%B2%D0%BE%D0%B5_%D0%B7%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5.ipynb)

Создание простого детектора овальных объектов. 

## Использование
Main.py позволяет протестировать модель. Скрипт получает на вход путь к исходному изображению и путь, куда нужно получить изображение после обработки нейросетью с указанными рамками овальных объектов. В процессе скрипт загружает предобученную модель из папки weights, делает предсказание и сохраняет исходное изображение с рамками. 
Перед использованием нужно:
1. Загрузить веса и скопировать их в папку weights [Google Drive](https://drive.google.com/drive/folders/15UBqKx4wQ9Axi9x8ZS-CxnmjJJGCXOVA?usp=sharing)
2. Установить необходимые модули (Pillow, tensorflow 2.2.0, numpy)

Для использования необходимо выполнить команду: 
```python
python main.py -i "<path to input image>" -o "<path to output image>"
```
