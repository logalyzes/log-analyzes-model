from fastapi import FastAPI, BackgroundTasks, Query, HTTPException, Request
from .kafka_manager import KafkaManager
from .predictors import Predictor
from dotenv import load_dotenv
import os
from typing import List
from .trainer import Trainer


from .models.feedMessage import Message

load_dotenv()
MODEL_SAVE_NAME = "model.pt"
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC')
KAFKA_GROUP_ID = 'fastapi-group'

trainer = Trainer(model_path=MODEL_SAVE_NAME)
predictor = Predictor()
kafka_manager = KafkaManager(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,  trainer=trainer)
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    app.state.consumer = kafka_manager.create_consumer(group_id=KAFKA_GROUP_ID, topic_name=KAFKA_TOPIC)

    size = kafka_manager.get_topic_size(KAFKA_TOPIC)
    print(f"Topic Size : {size}")

    print("Starting Server")

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.consumer.close()

@app.get("/")
async def read_root():
    return {"message": \
        "This is log prediction service send your log to /predict enpoint with query parameter log,Example /predict?log=issue at file system"}

@app.get("/predict")
async def predict(log: str = Query(..., description="Log data for prediction")):
    res = predictor.predict(log)
    return {"prediction": res}


@app.get("/feed")
async def predict(messages: List[Message],background_tasks: BackgroundTasks):
    try:
        for message in messages:
            kafka_manager.produce_message(KAFKA_TOPIC, message.model_dump())

        background_tasks.add_task(kafka_manager.train_model_if_needed, KAFKA_TOPIC)

        return {"status": "success", "message": "Messages produced successfully"}
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)