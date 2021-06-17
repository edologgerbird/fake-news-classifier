from flask_wtf import FlaskForm
from wtforms import (
    TextAreaField,
    SubmitField,
    IntegerField,
    SelectField
)
from wtforms.validators import (
    DataRequired,
)

class PredForm(FlaskForm):

    title = TextAreaField(
        'Article Title'
    )

    author = TextAreaField(
        'Article Author'
    )

    content = TextAreaField(
        'Article Content'
    )

    submit = SubmitField("Submit")