from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.on_event("startup")
async def startup():
    if not database.is_connected:
        await database.connect()
    # create a dummy entry
    await add_user()


@app.on_event("shutdown")
async def shutdown():
    if database.is_connected:
        await database.disconnect()


if __name__ == "__main__":  # pragma: no cover
    server = Server(
        Config(
            "runserver:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
        ),
    )
