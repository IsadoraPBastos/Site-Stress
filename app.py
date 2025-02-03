# pip install joblib / pip install numpy / pip install pandas / pip install scikit-learn

from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/previsao', methods=['GET', 'POST'])
def previsao():
    dadosono = request.form['sono']
    dadodordecabeca = request.form['dordecabeca']
    dadoperformance = request.form['performance']
    dadocarga = request.form['carga']
    dadoatividades = request.form['atividades']
    
    arr = np.array([dadosono, dadodordecabeca, dadoperformance, dadocarga, dadoatividades])
    
    arr = arr.astype(np.float64)
    
    pred = model.predict([arr])

    if pred == 1:
        nivel = "Estresse Baixo"
        msg = "Seu estresse está em um nível saudável. Continue assim! Mantenha uma rotina de autocuidado e descanso, para que esse nível não se eleve muito."
    elif pred == 2:
        nivel = "Estresse Leve"
        msg = "Você está um pouco estressado, mas nada que não muito preocupante ou que afete muito o seu corpo. Continue seguindo assim, mas sempre tente diminuir o nível de estresse o máximo possível."
    elif pred == 3:
        nivel = "Estresse Moderado"
        msg = "Seu estresse está um pouco acima do normal. É importante tomar medidas para evitar que ele aumente e se torne mais preocupante. Tente realizar mais atividades de lazer e descansar."
    elif pred == 4:
        nivel = "Estresse Alto"
        msg  = "Seu nível de estresse está elevado e precisa de atenção, pois isso pode afetar diretamente sua saúde. Invista em atividades de relaxamento, desenvolva novos hobbies e tente criar uma rotina de autocuidado. Avalie também se é possível diminuir a carga horária para conseguir mais tempo para descansar. "
    elif pred == 5:
        nivel = "Estresse Muito Alto"
        msg = "O seu estresse está em níveis muito elevados. É essecial que você desenvolva atividades de lazer e de descanso, assim como diminua a carga horária de estudo, ou isso trará sérios problemas de saúde ao seu corpo e a sua mente. Se lembre que o equilíbrio entre trabalho e descanso é necessário."
    
    return render_template('index.html', result=int(pred), msg=msg, nivel=nivel)

if __name__ == '__main__':
    app.run(debug=True)