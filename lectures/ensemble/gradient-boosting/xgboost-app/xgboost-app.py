# %%
import panel as pn
pn.extension('altair', 'tabulator')



# %%
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

pn.extension(sizing_mode="stretch_width", template="fast")
pn.state.template.param.update(site="Panel in the Browser", title="XGBoost Example")

iris_df = load_iris(as_frame=True)

trees = pn.widgets.IntSlider(start=2, end=30, name="Number of trees")

def pipeline(trees):
    model = XGBClassifier(max_depth=2, n_estimators=trees)
    model.fit(iris_df.data, iris_df.target)
    accuracy = round(accuracy_score(iris_df.target, model.predict(iris_df.data)) * 100, 1)
    return pn.indicators.Number(
        name="Test score",
        value=accuracy,
        format="{value}%",
        colors=[(97.5, "red"), (99.0, "orange"), (100, "green")],
    )

pn.Column(
    "Simple example of training an XGBoost classification model on the small Iris dataset.",
    iris_df.data.head(),
    "Move the slider below to change the number of training rounds for the XGBoost classifier. The training accuracy score will adjust accordingly.",
    trees,
    pn.bind(pipeline, trees),
).servable()


