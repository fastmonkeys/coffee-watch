import sys
import os
from datetime import datetime, timedelta

from flask import Flask, render_template
from flask.ext.sqlalchemy import SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URI', 'postgresql://localhost/coffee_watch'
)
db = SQLAlchemy(app)


class Measurement(db.Model):
    __tablename__ = 'measurement'
    id = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)


@app.route('/')
def hello_world():
    return render_template(
        'index.html',
        measurements=Measurement.query.order_by(Measurement.timestamp).all()
    )


if __name__ == "__main__":
    if '-create' in sys.argv:
        print("Creating tables")
        db.metadata.create_all(db.engine)
    elif '-dummy' in sys.argv:
        print("Creating dummy data")
        for i in range(100):
            db.session.add(Measurement(
                value=i,
                timestamp=datetime.now() - timedelta(minutes=100 - i)
            ))
        db.session.commit()

    else:
        print("Server start")
        try:
            app.run(host='0.0.0.0', port=8080)
        except KeyboardInterrupt:
            print("Shutdown")
