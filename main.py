import json
from fastapi import FastAPI, Depends
import crud
import predictive
import models
import schemas
from database import SessionLocal, engine
from sqlalchemy.orm import Session

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# maybe need to change int to uuid
@app.post("/create_user/")
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db=db, user=user)


# fix this and glucose_id we are incrementing automaticly
@app.post("/create_glucose/")
def create_glucose(glucose: schemas.GlucoseCreate, db: Session = Depends(get_db)):
    return crud.create_glucose(db=db, glucose = glucose)


# eating event incrementing automaticly
@app.post("/create_eating_event/", response_model=schemas.EatingEvent)
def create_eating_event(eating_event: schemas.EatingEventCreate, db: Session = Depends(get_db)):
    return crud.create_eating_event(db=db, eating_event=eating_event)


# insulin event incr automaticly
@app.post("/create_insulin_event/", response_model=schemas.InsulinEvent)
def create_insulin_event(insulin_event: schemas.InsulinEventCreate, db: Session = Depends(get_db)):
    return crud.create_insulin_event(db=db, insulin_event=insulin_event)

# eating event incrmenting auto
@app.post("/create_training_event/", response_model=schemas.TrainEvent)
def create_training_event(train_event: schemas.TrainEventCreate, db: Session = Depends(get_db)):
    return crud.create_train_event(db=db, train_event=train_event)


@app.get("/predict/{user_id}")
def get_predict(user_id: int, db: Session = Depends(get_db)):

    glucose = json.loads(crud.get_glucose_by_id(db=db, user_id=user_id))
    print(glucose)
    events = json.loads(crud.get_eating_event_by_id(db=db, user_id=user_id))
    print(events)
    insulin = json.loads(crud.get_insulin_event_by_id(db=db, user_id=user_id))
    print(insulin)
    return predictive.predict(user_id, {'glucose': glucose[::-1], 'events': events[::-1]})



# fit method
# @app.get("/fit/{user_id}")
# def get_fit(user_id: int, db: Session = Depends(get_db())):
#     glucose = json.loads(crud.get_all_glucose_by_id(db=db, user_id=user_id))
#     eating_event = json.loads(crud.get_eating_event_by_id(db=db, user_id=user_id))
#     insulin_event = json.loads(crud.get_insulin_event_by_id(db=db, user_id=user_id))
#     training_event = json.loads(crud.get_training_event_by_id(db=db, user_id=user_id))