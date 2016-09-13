import sys
import os
import json
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

    def __str__(self):
        return '<Measurement id={}, value={}, timestamp={}>'.format(self.id, self.value, self.timestamp)

    def __repr__(self):
        return '<Measurement id={}, value={}, timestamp={}>'.format(self.id, self.value, self.timestamp)


@app.route('/')
def hello_world():
    return render_template(
        'index.html',
        measurements=Measurement.query.order_by(Measurement.timestamp).all()
    )


@app.route('/api')
def api():
    measures = list(reversed(Measurement.query.order_by(Measurement.timestamp.desc()).limit(1000).all()[300:]))
    smoothed = [sorted(measures[i:i+5], key=lambda a: a.value)[2] for i in range(len(measures) - 5)]

    asdf = smoothed[0]
    brewed = smoothed[0]
    current = measures[-1]

    for item in smoothed:
        print(item.value, asdf.value)
        if item.value > 60 and asdf.value < 40:
            asdf = item
            brewed = item
            print('updated time')
        if item.value < 40:
            asdf = item

    return json.dumps({
        'current': current.value,
        'brewed_at': str(brewed.timestamp)
    })


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
