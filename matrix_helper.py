from ipywidgets import interact, FloatSlider, RadioButtons, widgets

def create_matrix_indexes_checker():
    w1 = widgets.Text(step=1, description='$a_{1, 2}$:', disabled=False)
    w2 = widgets.Text(step=1, description='$b_{1,0}$:', disabled=False)
    w3 = widgets.Text(step=1, description='$c_{0, 1}$:', disabled=False)
    w4 = widgets.Text(step=1, description='$d_{0, 0}$:', disabled=False)
    w5 = widgets.Text(step=1, description='$t_{0, 2}$:', disabled=False)

    w = [w1, w2, w3, w4, w5]
    for wid in w:
        display(wid)        

    result = [6, 5, 431, 4, 3534]      
    def check_matrix_incexes(wid, num):
        if wid.value == "":
            wid.value = "Значение не было введено"

        elif wid.value == str(num) + " Верно!":
            pass

        elif not wid.value.isdigit():
            wid.value = wid.value + " (пожалуйста, вводи только числа)"

        elif int(wid.value) == num:
            wid.value = wid.value + " Верно!"

        elif int(wid.value) != num:
            wid.value =  wid.value + " Не верно :("


    button = widgets.Button(description="Проверить ответы")
    display(button)

    def on_button_clicked(b):
        for wid, num in zip(w, result):
            check_matrix_incexes(wid, num)

    button.on_click(on_button_clicked)