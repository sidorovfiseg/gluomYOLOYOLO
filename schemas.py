import datetime
import uuid
from pydantic import BaseModel


class User(BaseModel):
    user_id: int
    diabetic_type: int

    class Config:
        orm_mode = True


class UserCreate(BaseModel):
    diabetic_type: int


class GlucoseCreate(BaseModel):
    user_id: int
    glucose_time: list[datetime.datetime]
    glucose_values: list[float]


class Glucose(BaseModel):
    glucose_id: int
    user_id: int
    glucose_time: list[datetime.datetime]
    glucose_values: list[float]

    class Config:
        orm_mode = True


class EatingEvent(BaseModel):
    eating_event_id: int
    user_id: int
    event_time: datetime.datetime
    calories: float
    proteins: float
    fats: float
    carbs: float

    class Config:
        orm_mode = True


class EatingEventCreate(BaseModel):
    user_id: int
    event_time: datetime.datetime
    calories: float
    proteins: float
    fats: float
    carbs: float


class InsulinEvent(BaseModel):
    insulin_event_id: int
    user_id: int
    event_time: datetime.datetime
    insulin_type: int
    insulin_amount: float

    class Config:
        orm_mode = True


class InsulinEventCreate(BaseModel):
    user_id: int
    event_time: datetime.datetime
    insulin_type: int
    insulin_amount: float


class TrainEvent(BaseModel):
    train_event_id: int
    user_id: int
    event_time: datetime.datetime
    train_duration: int
    train_type: int

    class Config:
        orm_mode = True


class TrainEventCreate(BaseModel):
    user_id: int
    event_time: datetime.datetime
    train_duration: int
    train_type: int


class PredictResponse(BaseModel):
    timestamp: datetime.datetime
    glucose: float

    class Config:
        orm_mode = True