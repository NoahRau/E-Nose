fmt:
    ruff check --fix
    ruff format

dump-influx DAYS OUTPUT_CSV_FILENAME:
	influx query \
		--host "${INFLUX_URL}" \
		--org "${INFLUX_ORG}" \
		--token "${INFLUX_TOKEN}" \
		--raw \
		'from(bucket: "'"${INFLUX_BUCKET}"'") \
		  |> range(start: -{{DAYS}}d) \
		  |> filter(fn: (r) => r._measurement == "sensor_metrics") \
		  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")' \
		> {{OUTPUT_CSV_FILENAME}}
