import databases
import ormar
import sqlalchemy
import datetime
from .config import settings
from typing import Optional
import json

database = databases.Database(settings.db_url)
metadata = sqlalchemy.MetaData()


class BaseMeta(ormar.ModelMeta):
    metadata = metadata
    database = database


class User(ormar.Model):
    class Meta(BaseMeta):
        tablename = "Users"

    id: int = ormar.Integer(primary_key=True, auto_increment=True)  
    name: str = ormar.Integer(max_length=100, nullable=False)                      
    email: str = ormar.String(max_length=100, unique=True, nullable=False)
    password: str = ormar.String(max_length=300, nullable=True, default="")    
    profile: str = ormar.String(max_length=10000, nullable=True, default="")     	    # can have this of token_google, but not both
    #token_google: str = ormar.String(max_length=1500, nullable=True, default="")        # can have this of password, but not both
    creted_at: datetime.datetime = ormar.DateTime(default=datetime.datetime.now)	    # set to crt datetime each time you call an endpoint
    #token: str = ormar.String(max_length=500, nullable=True, default="")

    def to_json(self, users):
        return {
            'id': self.id,
            'user_name': self.name,
            'user_password': self.password,
            'user_profile': self.profile,
        }
   


engine = sqlalchemy.create_engine(settings.db_url)
metadata.create_all(engine)