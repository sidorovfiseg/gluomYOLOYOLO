import json

from sqlalchemy.orm import Session
import models
import schemas
from sqlalchemy import desc


def create_user(db: Session, user: schemas.User):
    db_user = models.User(diabetic_type=user.diabetic_type)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user.user_id


def create_glucose(db: Session, glucose: schemas.Glucose):
    
    for i in range(min(len(glucose.glucose_values), len(glucose.glucose_time))):
    
        db_glucose = models.Glucose(user_id=glucose.user_id,
                                    glucose_time=glucose.glucose_time[i],
                                    glucose_value=glucose.glucose_values[i])
        db.add(db_glucose)
        db.commit()
        db.refresh(db_glucose)
    return db_glucose


def create_train_event(db: Session, train_event: schemas.TrainEvent):
    db_train_event = models.TrainEvent(user_id=train_event.user_id,
                                       event_time=train_event.event_time,
                                       train_duration=train_event.train_duration,
                                       train_type=train_event.train_type)
    db.add(db_train_event)
    db.commit()
    db.refresh(db_train_event)
    return db_train_event


def create_insulin_event(db: Session, insulin_event: schemas.InsulinEvent):
    db_insulin_event = models.InsulinEvent(user_id=insulin_event.user_id,
                                           event_time=insulin_event.event_time,
                                           insulin_type=insulin_event.insulin_type,
                                           insulin_amount=insulin_event.insulin_amount)
    db.add(db_insulin_event)
    db.commit()
    db.refresh(db_insulin_event)
    return db_insulin_event


def create_eating_event(db: Session, eating_event: schemas.EatingEvent):
    db_eating_event = models.EatingEvent(user_id=eating_event.user_id,
                                         event_time=eating_event.event_time,
                                         calories=eating_event.calories,
                                         proteins=eating_event.proteins,
                                         fats=eating_event.fats,
                                         carbs=eating_event.carbs)
    db.add(db_eating_event)
    db.commit()
    db.refresh(db_eating_event)
    return db_eating_event


def get_glucose_by_id(db: Session, user_id: int):
    result = db.execute(db.query(models.Glucose.glucose_time, models.Glucose.glucose_value).
                        filter(models.Glucose.user_id == user_id).order_by(desc(models.Glucose.glucose_id)).limit(20))
    rows = [{'datetime': i, 'Gl': j} for i, j in result.fetchall()]
    return json.dumps(rows, default=str)


def get_event_by_id(db: Session, user_id: int):
    result = db.execute(db.query(models.EatingEvent.event_time,
                                 models.EatingEvent.calories,
                                 models.EatingEvent.proteins,
                                 models.EatingEvent.fats,
                                 models.EatingEvent.carbs).
                        filter(models.EatingEvent.user_id == user_id).\
                        order_by(desc(models.EatingEvent.eating_event_id)).limit(20))
    rows = [{'datetime': i, 'k': j, 'b': k, 'j': l, 'u': p} for i, j, k, l, p in result.fetchall()]
    return json.dumps(rows, default=str)

