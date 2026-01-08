fmt:
	black .
	isort .

dump-influx DAYS=7 FILE=export.csv:
	influx query \
		--host "${INFLUX_URL}" \
		--org "${INFLUX_ORG}" \
		--token "${INFLUX_TOKEN}" \
		--raw \
		'from(bucket: "'"${INFLUX_BUCKET}"'") \
		  |> range(start: -{{DAYS}}d) \
		  |> filter(fn: (r) => r._measurement == "sensor_metrics") \
		  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")' \
		> {{FILE}}
