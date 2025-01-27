from flask import Flask,render_template
import lab4

app = Flask(__name__)

@app.route('/')
def index():
    # per1 = lab4.AlgoritmoPerceptron(0.1, 0.6, 10, 1, 5, -6, 4, -2, 2, -1, 0, 4, -4)
    # r1, r2 = per1.generar_puntos_aleatorios_en_dos_regiones()
    # per1.set_patrones_salidas(r1, r2)
    # per1.set_w_b_ln()
    # per1.perceptron()
    return render_template('index.html')
    #return f"El resultado es: {per1.perceptron}"

if __name__ == '__main__':
    app.run(debug=True,port=5000)