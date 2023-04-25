import datetime

from sqlalchemy import Column, ForeignKey, \
    Integer, Float, DateTime, BigInteger


from database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    diabetic_type = Column(Integer, nullable=False)


class Glucose(Base):
    __tablename__ = "glucose"

    glucose_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    glucose_time = Column(DateTime)
    glucose_value = Column(Float)


class EatingEvent(Base):
    __tablename__ = "eating_events"

    eating_event_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(BigInteger, ForeignKey("users.user_id"))
    event_time = Column(DateTime)
    calories = Column(Float)
    proteins = Column(Float)
    fats = Column(Float)
    carbs = Column(Float)


class InsulinEvent(Base):
    __tablename__ = "insulin_events"

    insulin_event_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    event_time = Column(DateTime)
    insulin_type = Column(Integer)
    insulin_amount = Column(Float)


class TrainEvent(Base):
    __tablename__ = "train_events"

    train_event_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    event_time = Column(DateTime)
    train_duration = Column(Integer)
    train_type = Column(Integer)
