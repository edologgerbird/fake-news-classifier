from flask import Flask, render_template, request, url_for, session
from werkzeug.utils import redirect
from form import PredForm
import predictor
import os 

app = Flask(__name__, template_folder='templates') 

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


@app.route("/", methods=["GET", "POST"])
def pred_form():
    form = PredForm()
    if form.validate_on_submit():

        session['title'] = form.title.data
        session['author'] = form.author.data
        session['content'] = form.content.data
        x_input = predictor.format_input(session['title'],session['author'],session['content'])
        preds = predictor.predict(x_input)
        session['pred_label'] = str(preds[0])
        session['pred_0'] = str(preds[1])
        session['pred_1'] = str(preds[2])

        return redirect(url_for("success"))
    return render_template(
        "predform.jinja2",
        form=form,
        template="form-template"
    )

@app.route("/success", methods=["GET", "POST"])
def success():
    """Generic success page upon form submission."""
    return render_template(
        "success.jinja2",
        template="success-template",
        pred_1=session['pred_1'],
        pred_0=session['pred_0'],
        pred_label=session['pred_label']
    )

if __name__ == '__main__':
    app.run(debug=True)