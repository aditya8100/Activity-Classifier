let express = require("express")
const app = express()

var current = 'F'
app.listen(process.env.PORT || 8080, '0.0.0.0', () => console.log('webhook is listening'));

app.get('/status', (req, res) => {
    res.status(200).send(current)
})