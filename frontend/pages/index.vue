<script lang="ts">
import {defineComponent} from 'vue'
import '~/assets/css/index.css'

const BASE_URL = "http://127.0.0.1:8000"


export default defineComponent({
  name: "index",
  components: {  },
  data() {
    return {
      form: {
        district: "Kicukiro",
        province: "Kigali_City"
      },
      districtNames: ['Kicukiro', 'Gasabo', 'Nyarugenge'],
      allDistrictNames: ['Gatsibo', 'Muhanga', 'Gicumbi', 'Kicukiro', 'Rulindo', 'Nyaruguru',
       'Rwamagana', 'Kamonyi', 'Gasabo', 'Ngororero', 'Rubavu', 'Nyamagabe',
       'Burera', 'Nyabihu', 'Gakenke', 'Nyamasheke', 'Kirehe', 'Nyagatare',
       'Bugesera', 'Karongi', 'Huye', 'Musanze', 'Rusizi', 'Ngoma',
       'Nyarugenge', 'Gisagara', 'Kayonza', 'Ruhango', 'Nyanza', 'Rutsiro'
      ],
      provinces: {Eastern_Province: ['Gatsibo', 'Rwamagana', 'Kirehe', 'Nyagatare', 'Bugesera', 'Ngoma', 'Kayonza'],
       Southern_Province: ['Muhanga', 'Nyaruguru', 'Kamonyi', 'Nyamagabe', 'Huye', 'Gisagara', 'Ruhango', 'Nyanza'],
       Northern_Province: ['Gicumbi', 'Rulindo', 'Burera', 'Gakenke', 'Musanze'],
       Kigali_City: ['Kicukiro', 'Gasabo', 'Nyarugenge'],
       Western_Province: ['Ngororero', 'Rubavu', 'Nyabihu', 'Nyamasheke', 'Karongi', 'Rusizi', 'Rutsiro'
      ]},
      seasonal_data: [
        {
          x: [],
          y: [],
          type: 'scatter',
          mode: 'lines',
          name: 'Time Series Data'
        }
      ],
      annual_data: [
        {
          x: [],
          y: [],
          type: 'scatter',
          mode: 'lines',
          name: 'Time Series Data'
        }
      ],
      seasonal_layout: {
        title: 'Rainfall Forecast for Season A',
        xaxis: {
          title: 'Date'
        },
        yaxis: {
          title: 'Weather',
          tickvals: [0, 1, 2, 3], // Specify the discrete values (0 and 1)
          ticktext: ['Intermediary', 'Season A', 'Season C', 'Season B'] // Specify the corresponding labels
        }
      },
      annual_layout: {
        title: 'Annual Rainfall Forecast',
        xaxis: {
          title: 'Date'
        },
        yaxis: {
          title: 'Weather',
          tickvals: [0, 1, 2, 3], // Specify the discrete values (0 and 1)
          ticktext: ['Intermediary', 'Season A', 'Season C', 'Season B'] // Specify the corresponding labels
        }
      }

    }
  },
  async mounted() {
    const res_seasonal = await this.getForecast('season_a/single_district/',
      { district: this.form.district })
    const res_annual = await this.getForecast('single_district/',
        { district: this.form.district })

    console.log('Seasonal data: ', res_seasonal)

    this.seasonal_data[0].x = res_seasonal.date
    this.seasonal_data[0].y = res_seasonal.states

    this.annual_data[0].x = res_annual.date
    this.annual_data[0].y = res_annual.states
    this.annual_layout.title = `Annual Rainfall Forecast in ${this.form.district}, ${this.form.province.replace('_', ' ')}`
    this.seasonal_layout.title = `Rainfall Forecast for Season A in ${this.form.district}, ${this.form.province.replace('_', ' ')}`

    // // Create the plot
    Plotly.newPlot('seasonal-forecast', this.seasonal_data, this.seasonal_layout);
    Plotly.newPlot('annual-forecast', this.annual_data, this.annual_layout);
  },

  setup() {
    // const { data: value } = useFetch(`${BASE_URL}/season_a/province/`)
  },

  methods: {
    async getForecast(path: string, payload: Object) {
      const res = await $fetch(`${BASE_URL}/${path}`, {
        headers: {
          Accept: "application/json"
        },
        method: 'POST',
        body: payload
      })

      return {
        date: Object.keys(res.predictions.hidden_states),
        states: Object.values(res.predictions.hidden_states)
      }
    },

    async filterForecast() {
      const res_seasonal = await this.getForecast('season_a/single_district/',
        { district: this.form.district })
      const res_annual = await this.getForecast('single_district/',
          { district: this.form.district })

      this.annual_layout.title = `Annual Rainfall Forecast in ${this.form.district}, ${this.form.province.replace('_', ' ')}`
      this.seasonal_layout.title = `Rainfall Forecast for Season A in ${this.form.district}, ${this.form.province.replace('_', ' ')}`

      this.seasonal_data[0].x = res_seasonal.date
      this.seasonal_data[0].y = res_seasonal.states

      this.annual_data[0].x = res_annual.date
      this.annual_data[0].y = res_annual.states

      Plotly.newPlot('seasonal-forecast', this.seasonal_data, this.seasonal_layout);
      Plotly.newPlot('annual-forecast', this.annual_data, this.annual_layout);
    },
  },
  watch: {
    "form.province" (v) {
      if (!v || v.length === 0)
        this.districtNames = this.allDistrictNames
      else {
        if (v === "Kigali_City")
          this.districtNames = this.provinces.Kigali_City
        else if (v === "Eastern_Province")
          this.districtNames = this.provinces.Eastern_Province
        else if (v === "Western_Province")
          this.districtNames = this.provinces.Western_Province
        else if (v === "Northern_Province")
          this.districtNames = this.provinces.Northern_Province
        else
          this.districtNames = this.provinces.Southern_Province
    }
  }
  }

})
</script>

<template>
 <div class="container main-wrapper d-flex flex-column justify-items-center 100vh">
   <div class="row">
     <h1 class="text-center mt-4">Seasonal Rainfall Forecasts for Different Sectors in Rwanda</h1>
   </div>
   <div class="row flex-grow-1">
     <div class="col-4 d-flex flex-column align-content-center justify-content-center">
       <h4>Choose Province and Corresponding District</h4>

       <form @submit.prevent="filterForecast">
         <div class="form-floating mb-3 mt-3">
           <select class="form-select form-control" id="province" name="province" v-model="form.province">
              <option value="Kigali_City">Kigali City</option>
              <option value="Eastern_Province">Eastern</option>
              <option value="Western_Province">Western</option>
              <option value="Northern_Province">Northern</option>
             <option value="Southern_Province">Southern</option>
            </select>
            <label for="province">Province</label>
          </div>

          <div class="form-floating mt-3 mb-3">
            <select class="form-select form-control" id="district" name="district" v-model="form.district">
              <option :value="district" v-for="(district, i) in districtNames" :key="i">{{district}}</option>
            </select>
            <label for="district">District</label>
          </div>

         <input type="submit" value="GET FORECASTS" class="button btn-secondary btn-block p-2 w-100"
                :disabled="form.district.length === 0 || form.province.length === 0">
       </form>
     </div>
     <div class="col-8 d-flex flex-column h-100">
       <div class="d-flex flex-column justify-content-center align-content-center flex-grow-1 graphs">

         <div id="seasonal-forecast" class="flex-fill w-100 h-50"></div>

         <div id="annual-forecast" class="flex-fill w-100 h-50"></div>

       </div>

     </div>
   </div>
 </div>
</template>

<style scoped>
@import "assets/css/index.css";

</style>