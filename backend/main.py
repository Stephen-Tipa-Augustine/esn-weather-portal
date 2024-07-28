from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Optional
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Define the origins that are allowed to access the API
origins = ["*"]

# Add the CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define RainDataLoader class for loading data
class RainDataLoader:
    def __init__(self, data):
        self.data = data

    def read_province_data(self, province=None):
        if province:
            return self.data[self.data['province'] == province]
        else:
            return self.data
    def read_district_data(self, district=None):
        if district:
            return self.data[self.data['district'] == district]
        else:
            return self.data


# Load the training data
df_rainfall = pd.read_csv('training_data.csv')
df_rainfall['Unnamed: 0'] = pd.to_datetime(df_rainfall['Unnamed: 0'])
df_rainfall.set_index('Unnamed: 0', drop=True, inplace=True)
df_rainfall.index = pd.DatetimeIndex(df_rainfall.index)

# Initialize the data loader and hmm model
data_loader = RainDataLoader(df_rainfall)

# Load the trained hmm model using pickle
with open("models/rwanda.pkl", "rb") as file:
    hmm_model = pickle.load(file)

# Define request model
class ProvinceRequest(BaseModel):
    province: Optional[str] = None

# Define request model
class DistrictRequest(BaseModel):
    district: Optional[str] = None

# Define forecast start and end dates
forecast_start_date = pd.Timestamp('2024-01-01 00:00:00', tz='UTC')
# forecast_end_date = pd.Timestamp('2024-08-20 00:00:00', tz='UTC')


#############################################################################
#ALL SEASONS
#############################################################################
@app.post("/single_district/")
async def get_single_district(request: DistrictRequest):
    try:
        # Load the HMM model corresponding to the requested district
        model_path = f"models/districts/{request.district}.pkl"
        with open(model_path, "rb") as file:
            hmm_model = pickle.load(file)

        forecast_states, _ = hmm_model.sample(n_samples=366)
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=pd.date_range(start=forecast_start_date, periods=366)
        )
        
        return {"predictions": data.to_dict()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found for the specified district.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/province/")
async def get_province_data(request: ProvinceRequest):
    try:
        # Load the HMM model corresponding to the requested province
        model_path = f"models/provinces/{request.province}.pkl"
        with open(model_path, "rb") as file:
            hmm_model = pickle.load(file)

        forecast_states, _ = hmm_model.sample(n_samples=366)
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=pd.date_range(start=forecast_start_date, periods=366)
        )
        
        return {"predictions": data.to_dict()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found for the specified province.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/all_districts/")
async def get_all_districts_data():
    try:
        # Forecast future rainfall intensity levels
        forecast_states, _ = hmm_model.sample(n_samples=366)
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=pd.date_range(start=forecast_start_date, periods=366)
        )
        
        return {"predictions": data.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

#############################################################################
#SEASON A
#############################################################################

season_a_forecast_start_date = pd.Timestamp('2024-01-01 00:00:00', tz='UTC')

@app.post("/season_a/single_district/")
async def get_single_district(request: DistrictRequest):
    try:
        # Load the HMM model corresponding to the requested district
        model_path = f"models/districts/{request.district}.pkl"
        with open(model_path, "rb") as file:
            hmm_model = pickle.load(file)

        index = pd.date_range(start=season_a_forecast_start_date, end=pd.Timestamp('2024-06-30 00:00:00', tz='UTC'))
        forecast_states, _ = hmm_model.sample(n_samples=len(index))
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=index
        )
        
        return {"predictions": data.to_dict()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found for the specified district.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/season_a/province/")
async def get_province_data(request: ProvinceRequest):
    try:
        # Load the HMM model corresponding to the requested province
        model_path = f"models/provinces/{request.province}.pkl"
        with open(model_path, "rb") as file:
            hmm_model = pickle.load(file)

        forecast_states, _ = hmm_model.sample(n_samples=122)
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=pd.date_range(start=season_a_forecast_start_date, periods=122)
        )
        
        return {"predictions": data.to_dict()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found for the specified province.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/season_a/all_districts/")
async def get_all_districts_data():
    try:
        # Forecast future rainfall intensity levels
        forecast_states, _ = hmm_model.sample(n_samples=122)
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=pd.date_range(start=season_a_forecast_start_date, periods=122)
        )
        
        return {"predictions": data.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

#############################################################################
#SEASON B
#############################################################################

season_b_forecast_start_date = pd.Timestamp('2024-07-01 00:00:00', tz='UTC')

@app.post("/season_b/single_district/")
async def get_single_district(request: DistrictRequest):
    try:
        # Load the HMM model corresponding to the requested district
        model_path = f"models/districts/{request.district}.pkl"
        with open(model_path, "rb") as file:
            hmm_model = pickle.load(file)

        forecast_states, _ = hmm_model.sample(n_samples=62)
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=pd.date_range(start=season_b_forecast_start_date, periods=62)
        )
        
        return {"predictions": data.to_dict()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found for the specified district.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/season_b/province/")
async def get_province_data(request: ProvinceRequest):
    try:
        # Load the HMM model corresponding to the requested province
        model_path = f"models/provinces/{request.province}.pkl"
        with open(model_path, "rb") as file:
            hmm_model = pickle.load(file)

        forecast_states, _ = hmm_model.sample(n_samples=62)
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=pd.date_range(start=season_b_forecast_start_date, periods=62)
        )
        
        return {"predictions": data.to_dict()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found for the specified province.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/season_b/all_districts/")
async def get_all_districts_data():
    try:
        # Forecast future rainfall intensity levels
        forecast_states, _ = hmm_model.sample(n_samples=122)
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=pd.date_range(start=season_b_forecast_start_date, periods=122)
        )
        
        return {"predictions": data.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

#############################################################################
#SEASON C
#############################################################################

season_c_forecast_start_date = pd.Timestamp('2024-09-01 00:00:00', tz='UTC')

@app.post("/season_c/single_district/")
async def get_single_district(request: DistrictRequest):
    try:
        # Load the HMM model corresponding to the requested district
        model_path = f"models/districts/{request.district}.pkl"
        with open(model_path, "rb") as file:
            hmm_model = pickle.load(file)

        forecast_states, _ = hmm_model.sample(n_samples=181)
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=pd.date_range(start=season_c_forecast_start_date, periods=181)
        )
        
        return {"predictions": data.to_dict()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found for the specified district.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/season_c/province/")
async def get_province_data(request: ProvinceRequest):
    try:
        # Load the HMM model corresponding to the requested province
        model_path = f"models/provinces/{request.province}.pkl"
        with open(model_path, "rb") as file:
            hmm_model = pickle.load(file)

        forecast_states, _ = hmm_model.sample(n_samples=181)
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=pd.date_range(start=season_c_forecast_start_date, periods=181)
        )
        
        return {"predictions": data.to_dict()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found for the specified province.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/season_c/all_districts/")
async def get_all_districts_data():
    try:
        # Forecast future rainfall intensity levels
        forecast_states, _ = hmm_model.sample(n_samples=181)
        forecast_hidden_states = hmm_model.predict(forecast_states)

        # Create a DataFrame with the forecasted hidden states and states
        data = pd.DataFrame(
            data={
                'hidden_states': forecast_hidden_states
                # 'forecast_states': forecast_states.flatten()
            },
            index=pd.date_range(start=season_c_forecast_start_date, periods=181)
        )
        
        return {"predictions": data.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))