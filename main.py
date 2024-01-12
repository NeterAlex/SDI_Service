from fastapi import FastAPI

from routers import calculator_router

app = FastAPI()

app.include_router(calculator_router)


@app.get("/ping")
async def root():
    return {"message": "pong"}
