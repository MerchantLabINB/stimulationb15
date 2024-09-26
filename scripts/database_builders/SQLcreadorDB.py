# SQLcreadorDB.py
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, Date, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Crear el motor de la base de datos
engine = create_engine('sqlite:///microstimulation.db')
Base = declarative_base()

# Definición de las clases
class Experimento(Base):
    __tablename__ = 'experimentos'
    id = Column(Integer, primary_key=True, autoincrement=True)
    dia_experimental = Column(Date)
    sujeto_experimental = Column(String)
    peso = Column(Float)
    impedancia_electrodo = Column(Text)
    coordenadas = relationship('Coordenada', back_populates='experimento')
    videos = relationship('Video', back_populates='experimento')
    estimulos = relationship('Estimulo', back_populates='experimento')

class Coordenada(Base):
    __tablename__ = 'coordenadas'
    id = Column(Integer, primary_key=True, autoincrement=True)
    experimento_id = Column(Integer, ForeignKey('experimentos.id'))
    coordenada_x = Column(Float)
    coordenada_y = Column(Float)
    distancia_al_tejido = Column(Float)
    vueltas_debajo_del_grid = Column(Float)
    distancia_descendida = Column(Float)
    distancia_intracortical = Column(Float)
    canula_expuesta = Column(Float)
    experimento = relationship('Experimento', back_populates='coordenadas')
    videos = relationship('Video', back_populates='coordenada')
    estimulos = relationship('Estimulo', back_populates='coordenada')

class Video(Base):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True, autoincrement=True)
    experimento_id = Column(Integer, ForeignKey('experimentos.id'))
    coordenada_id = Column(Integer, ForeignKey('coordenadas.id'))
    angle = Column(String)
    hora = Column(DateTime)
    archivos_video = Column(Text)
    experimento = relationship('Experimento', back_populates='videos')
    coordenada = relationship('Coordenada', back_populates='videos')

class Estimulo(Base):
    __tablename__ = 'estimulos'
    id = Column(Integer, primary_key=True, autoincrement=True)
    experimento_id = Column(Integer, ForeignKey('experimentos.id'))
    coordenada_id = Column(Integer, ForeignKey('coordenadas.id'))
    hora = Column(DateTime)
    ensayos = Column(Integer)
    ensayos_movimiento_evocado = Column(Integer)
    amplitud = Column(Float)
    duracion = Column(Float)
    forma_del_pulso = Column(Text)
    frecuencia = Column(Float)
    top_de_la_corteza = Column(Float)
    profundidad_electrodo = Column(Float)
    archivos_video = Column(Text)
    archivo_del_estimulo_descargado = Column(Text)
    frames = relationship('Frame', back_populates='estimulo')
    experimento = relationship('Experimento', back_populates='estimulos')
    coordenada = relationship('Coordenada', back_populates='estimulos')

class Frame(Base):
    __tablename__ = 'frames'
    id = Column(Integer, primary_key=True, autoincrement=True)
    estimulo_id = Column(Integer, ForeignKey('estimulos.id'))
    start_frame_lateral = Column(Integer)
    end_frame_lateral = Column(Integer)
    start_frame_frontal = Column(Integer, nullable=True)
    segundo_equivalente_inicio_lateral = Column(Float)
    tiempo_equivalente_fin_lateral = Column(Float)
    segundo_equivalente_frontal = Column(Float, nullable=True)
    estimulo = relationship('Estimulo', back_populates='frames')

# Crear las tablas en la base de datos
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

print("Tablas creadas correctamente.")
