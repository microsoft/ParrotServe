# Parrot APIs w/ Semantic Variable

Parrot provides OpenAI-like APIs with the extension of Semantic Variables. As [Semantic Variables Design](../sys_design/app_layer/semantic_variable.md)

### Session

Endpoint: `/{api_version}/session`

- Register a session in OS. [POST]

Request body:

```json
{
	"api_key": "xxx" // User's API Key
	"apps": [
		{"app_id": "xxx"},
	],
}
```

Response  body:

```json
{
	"session_id": "xxx",
	"session_auth": "yyy"
}
```

- Delete a session in OS. [DELETE]

Request body:

```json
{
	"session_id": "xxx",
	"session_auth": "yyy"
}
```

Response body:

```json
{}
```

NOTE: A session will expire in 12H (Charging user according to time?).

### Submit Semantic Function Call

Endpoint: `/{api_version}/semantic_call`

- Submit a semantic function call request. [POST]

Request body:

```json
{
	"template": "This is a test {{a}} function. {{b}}",
	"placeholders": [
		{
			"name": "a",
			"is_output": false / true,
			"var_id": "bbb", // Optional
			"sampling_config": {
	      "temperature": "xxx",
	      "top_p": "xxx",
	    }
		},
	],
	"session_id": "xxx",
	"session_auth": "yyy",
  "models": ["model1", "model2", ...] // Optional. If specified, the request will be scheduled only to these models. By default ([]) it can be scheduled to any model.
	"model_type": "token_id / text",
	"remove_pure_fill": true / false,
}
```

Response body:

```json
{
	"request_id": "xxx",
	"session_id": "yyy",
	"created_vars": [
			{
					"placeholder_name": "fff",
					"is_output": true / false,
			    "var_name": "ddd",
			    "var_id": "ccc",
			    "var_desc": "The first output of request xxx",
					"var_scope": "eeee",
			}
	]
}
```

### Submit Native Function Call

> NOTE: This API is not stable

We have some built-in native functions. We don’t allow user to submit their customized code because it may introduce safety problems.

Endpoint: `/{api_version}/native_call`

- Submit a native call [POST].

Request body:

```json
{
	"func_name": "xxx",
	... // (the detailed args are dependent to the native function called)
}
```


### Semantic Variable

The semantic variable object.

```json
{
	"var_id": "ccc",
  "var_name": "ddd",
  "var_desc": "The first output of request xxx",
	"var_scope": "eeee",
}
```

Endpoint: `/{api_version}/semantic_var/`

- Get a full list of semantic variables in current scope.

Request body:

```json
{
	"session_id": "xxx",
	"session_auth": "yyy",
}
```

Response body: Returns a list of vars.

```json
{
	"vars": [
		{
			"var_id": "ccc",
			"var_name": "ddd",
			"var_desc": "The first output of request xxx",
			"var_scope": "eeee",
		},
		...
	]
}
```

Endpoint:  `/{api_version}/semantic_var/{var_id}` 

- Get a value of a semantic variable. [GET]

Request body:

```json
{
	"session_id": "xxx",
	"session_auth": "yyy",
	"criteria": "zzz"
}
```

Response body:

```json
{
	"content": "zzz"
}
```

- Set a value of a semantic variable. [POST]

Request body:

```json
{
	"session_id": "xxx",
	"session_auth": "yyy",
	"content": "zzz",
}
```

Response body:

```json
{}
```

### Models

Endpoint: `/{api_version}/models`

- Get a list of supported model names in the system. [GET]

Request body:

```json
{
	"session_id": "xxx",
	"session_auth": "yyy",
}
```

Response body:

```json
{
	"models": [
		{
			"model_name": "xxx",
			"tokenizer_name": "yyy"
		}
	]
}
```