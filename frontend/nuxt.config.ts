// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: true },
   app: {
    head: {
        link: [{ rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css' }],
        script: [
             {
                src: "https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js",
                async: true,
                crossorigin: "anonymous",
                type: 'text/javascript'
            },
            {
                src: "https://cdn.plot.ly/plotly-latest.min.js",
                async: true,
                crossorigin: "anonymous",
                type: 'text/javascript'
            },
            // {
            //     src: "https://cdn.jsdelivr.net/npm/chart.js",
            //     async: true,
            //     crossorigin: "anonymous",
            //     type: 'text/javascript'
            // },
            // {
            //     src: "https://cdn.jsdelivr.net/npm/chart.js/dist/chart.min.js",
            //     async: true,
            //     crossorigin: "anonymous",
            //     type: 'text/javascript'
            // }
        ]
      }
    }
})
