from crypt import methods
from types import MethodType
from flask import Flask, request
from flask import render_template, redirect, url_for
from wtforms import SubmitField, FileField, IntegerField,FloatField
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired, NumberRange
from flask_bootstrap import Bootstrap
import model
from model import Params_GB, Params_RF
import os

app = Flask(__name__)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)

class GradientBoostingParams(FlaskForm):
    n_estimators = IntegerField("n_estimators", validators=[DataRequired(), NumberRange(1, 1000)])
    max_depth = IntegerField("max_depth(0 to unlimited)", validators=[DataRequired(), NumberRange(0, 1000)])
    feature_subsample_size = IntegerField("feature_subsample_size(0 to unstated)",
                                          validators=[DataRequired(), NumberRange(0)])
    learning_rate = FloatField("learning_rate", validators=[DataRequired(), NumberRange(0)])
    file_path = FileField('Path', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Train model')

class RandomForestParams(FlaskForm):
    n_estimators = IntegerField("n_estimators", validators=[DataRequired(), NumberRange(1, 1000)])
    max_depth = IntegerField("max_depth", validators=[DataRequired(), NumberRange(0, 1000)])
    feature_subsample_size = IntegerField("feature_subsample_size",
                                          validators=[DataRequired(), NumberRange(0)])
    file_path = FileField('Path', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Train model')

class TestDataset(FlaskForm):
    file_path = FileField('Test dataset', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Get result')

params_RF = Params_RF()
params_GB = Params_GB()
train_dataset_path_GB = None
test_dataset_path_GB = None
train_dataset_path_RF = None
test_dataset_path_RF = None


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return render_template('new_model.html')

@app.route('/GradientBoosting', methods=['GET', 'POST'])
def gradient_boosting_init():
    global train_dataset_path_GB, params_GB
    gradient_boosting_form = GradientBoostingParams()
    if gradient_boosting_form.validate_on_submit():
        train_dataset_path_GB = gradient_boosting_form.file_path.data.filename
        file = gradient_boosting_form.file_path.data
        if not os.path.isdir("instance"):
            os.mkdir("instance")
        file.save(os.path.join(app.instance_path, train_dataset_path_GB))
        params_GB.n_estimators = gradient_boosting_form.n_estimators.data
        params_GB.max_depth = gradient_boosting_form.max_depth.data
        params_GB.feature_subsample_size = gradient_boosting_form.feature_subsample_size.data
        params_GB.learning_rate = gradient_boosting_form.learning_rate.data
        return redirect('/trained_gradient_boosting')
    return render_template('from_form.html', form=gradient_boosting_form)

@app.route('/trained_gradient_boosting', methods=['GET', 'POST'])
def trained_gradient_boosting():
    global test_dataset_path_GB
    try:
        model.train_gb(params_GB, train_dataset_path_GB)
    except:
        return redirect('/error')
    test_dataset_form = TestDataset()
    if test_dataset_form.validate_on_submit():
        test_dataset_path_GB = test_dataset_form.file_path.data.filename
        file = test_dataset_form.file_path.data
        if not os.path.isdir("instance"):
            os.mkdir("instance")
        file.save(os.path.join(app.instance_path, test_dataset_path_GB))
        return redirect('test_result_GB')
    return render_template('trained_GB.html', params=params_GB,
                           form=test_dataset_form, path=train_dataset_path_GB)

@app.route('/test_result_GB', methods=['GET'])
def test_gradient_boosting():
    try:
        ans = list(model.test_gb(test_dataset_path_GB))
    except:
        return redirect('/error')
    return render_template('test_result.html', ans=ans)

@app.route('/RandomForest', methods=['GET', 'POST'])
def random_forest_init():
    global train_dataset_path_RF, params_RF
    random_forest_form = RandomForestParams()
    if random_forest_form.validate_on_submit():
        train_dataset_path_RF = random_forest_form.file_path.data.filename
        file = random_forest_form.file_path.data
        if not os.path.isdir("instance"):
            os.mkdir("instance")
        file.save(os.path.join(app.instance_path, train_dataset_path_RF))
        params_RF.n_estimators = random_forest_form.n_estimators.data
        params_RF.max_depth = random_forest_form.max_depth.data
        params_RF.feature_subsample_size = random_forest_form.feature_subsample_size.data
        return redirect('/trained_random_forest')
    return render_template('from_form.html', form=random_forest_form)

@app.route('/trained_random_forest', methods=['GET', 'POST'])
def train_random_forest():
    global test_dataset_path_RF
    try:
        model.train_rf(params_RF, train_dataset_path_RF)
    except:
        return redirect('/error')
    test_dataset_form = TestDataset()
    if test_dataset_form.validate_on_submit():
        test_dataset_path_RF = test_dataset_form.file_path.data.filename
        file = test_dataset_form.file_path.data
        if not os.path.isdir("instance"):
            os.mkdir("instance")
        file.save(os.path.join(app.instance_path, test_dataset_path_RF))
        return redirect('test_result_RF')
    return render_template('trained_RF.html', params=params_RF,
                           form=test_dataset_form, path=train_dataset_path_RF)

@app.route('/test_result_RF', methods=['GET'])
def test_random_forest():
    try:
        ans = list(model.test_rf(test_dataset_path_RF))
    except:
        return redirect('/error')
    return render_template('test_result.html', ans=ans)

@app.route('/error', methods=['GET'])
def error():
    return render_template('error.html')