import streamlit as st
from datasets import Dataset
from transformers import AutoTokenizer
from datasets import DatasetDict
from pathlib import Path
import torch
from transformers import AutoModelForQuestionAnswering, Trainer
from transformers import DataCollatorWithPadding
from unidecode import unidecode


# Necessary import for saving correctly
# (the error was found with summarization t5 small, after training).
import locale
def getpreferredencoding(do_setlocale=True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding


# Global path variables.
drive_path = "drive/MyDrive/tfg-juncodelasheras/"
colab_data_path = drive_path + "colab_saved_data/"

question_answering = "question_answering"
model_tinyroberta = "deepset/tinyroberta-squad2"

# This is the model that is currently used.
task_type = question_answering
checkpoint = model_tinyroberta

hub_checkpoint_cause = "mi_tinyROBERTA_cause"
hub_checkpoint_effect = "mi_tinyROBERTA_effect"


# Helper function of get_tokenized_datasets for question_answering task.
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    if 'answers' in examples:
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context.
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1

            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1

            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0).
            if offset[context_start][0] > end_char or \
               offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions.
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1

                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1

                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

    return inputs


# It gets the tokenized datasets from the given dataset(s).
def get_tokenized_datasets(datasets):
    # If datasets is instance of DatasetDict, it will contain train dataset.
    if isinstance(datasets, DatasetDict):
        column_names = datasets["train"].column_names
    # else it will be only one dataset.
    else:
        column_names = datasets.column_names

    tokenized_dataset = datasets.map(preprocess_function,
                                     batched=True,
                                     remove_columns=column_names)
    return tokenized_dataset


# If create_new_trainer is True, it will replace previous trainers.
def get_trainer(dataset_name,
                tokenizer,
                cause_model):
    trainer_name = dataset_name + "_trainer_" + task_type
    if cause_model:
        trainer_name += "_cause"
    elif not cause_model:
        trainer_name += "_effect"

    # model = AutoModelForQuestionAnswering \
    #    .from_pretrained(colab_data_path + checkpoint + "/" +
    #                     trainer_name)
    if cause_model:
        model = AutoModelForQuestionAnswering \
            .from_pretrained("Juncodh/" + hub_checkpoint_cause)
    elif not cause_model:
        model = AutoModelForQuestionAnswering \
            .from_pretrained("Juncodh/" + hub_checkpoint_effect)
    #    .from_pretrained(colab_data_path + checkpoint + "/" +
    #                     trainer_name)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer


# Normalize the string str so that the correct string can be compared
# with the predicted string with more precission.
def normalize_str(s):
    # Normalize left and right double quotes to standard double quotes.
    s = s.replace('”', '"').replace('“', '"')
    # If there are an odd number of character ",
    # the last one has to be removed to not have an error in the csv parser.
    if s.count('"') % 2 == 1:
        idx = s.rfind('"')
        s = s[:idx] + s[idx + 1:]
    # There are some strings that contain the separator character,
    # so it has to be removed.
    return unidecode(s.replace(';', '').replace('[CLS]', '')
                      .replace('[SEP]', '').replace('summarize:', '')
                      .lower()).strip()


# Helper function of get_predictions for question_answering task.
# cause_model is True if it is predicting the Cause, else
# it is predicting the Effect.
# Return [start predicted tokens], [end predicted tokens].
def get_predictions_question_answering(tokenized_datasets, cause_model):
    if cause_model:
        predictions, _, _ = trainer_cause.predict(tokenized_datasets)
    else:
        predictions, _, _ = trainer_effect.predict(tokenized_datasets)

    start_logits, end_logits = predictions

    logits_probabilities = torch.nn.functional \
                                   .softmax(torch.from_numpy(start_logits),
                                            dim=-1)
    start_predicted_token = logits_probabilities.argmax(dim=-1).tolist()

    logits_probabilities = torch.nn.functional \
                                   .softmax(torch.from_numpy(end_logits),
                                            dim=-1)
    end_predicted_token = logits_probabilities.argmax(dim=-1).tolist()

    return start_predicted_token, end_predicted_token


# Get the predictions of the cause and effect given the tokenized datasets.
def get_predictions():
    start_predicted_cause_token, end_predicted_cause_token = \
        get_predictions_question_answering(tokenized_cause_datasets, True)
    start_predicted_effect_token, end_predicted_effect_token = \
        get_predictions_question_answering(tokenized_cause_datasets, False)
    tokens_predicted_cause = \
        tokenized_cause_datasets[0]["input_ids"]\
                                [start_predicted_cause_token[0]:
                                 end_predicted_cause_token[0] + 1]
    tokens_predicted_effect = \
        tokenized_effect_datasets[0]["input_ids"]\
                                 [start_predicted_effect_token[0]:
                                  end_predicted_effect_token[0] + 1]
    str_predicted_cause = tokenizer.decode(tokens_predicted_cause)
    str_predicted_effect = tokenizer.decode(tokens_predicted_effect)
    str_predicted_cause = normalize_str(str_predicted_cause)
    str_predicted_effect = normalize_str(str_predicted_effect)
    return str_predicted_cause, str_predicted_effect


# It is necessary to cache the tokenizer and the trainers.
# Avoid getting "http://localhost:8501/script-health-check": EOF
@st.cache_resource
def load_cache():
    # It is the name of the dataset that was used when the model was trained.
    original_dataset_path = "dataset/training_subtask_es.csv"
    dataset_name = Path(original_dataset_path).stem

    # new_tokenizer_name = dataset_name + "_tokenizer_" + task_type
    # tokenizer = AutoTokenizer.from_pretrained(colab_data_path +
    #                                          checkpoint + "/" +
    #                                          new_tokenizer_name)

    tokenizer = AutoTokenizer.from_pretrained("Juncodh/" +
                                              hub_checkpoint_cause)

    trainer_name = dataset_name + "_trainer_" + task_type
    trainer_name += "_cause"
    trainer_cause = get_trainer(dataset_name, tokenizer, True)

    trainer_name = dataset_name + "_trainer_" + task_type
    trainer_name += "_effect"
    trainer_effect = get_trainer(dataset_name, tokenizer, False)
    return tokenizer, trainer_cause, trainer_effect


tokenized_cause_datasets = None
tokenized_effect_datasets = None


def predict_one_text(context_to_do_predictions):
    global tokenized_cause_datasets
    global tokenized_effect_datasets

    # Create the dataset with the given context.
    # Note, the question is always the same,
    # independently of the cause or effect dataset.
    dataset = {
        "context": [context_to_do_predictions],
        "question": ["Escribe la causa"],
    }

    cause_datasets = Dataset.from_dict(dataset)
    effect_datasets = Dataset.from_dict(dataset)

    tokenized_cause_datasets = get_tokenized_datasets(cause_datasets)
    tokenized_effect_datasets = get_tokenized_datasets(effect_datasets)

    str_predicted_cause, str_predicted_effect = get_predictions()
    return str_predicted_cause, str_predicted_effect


def main():
    # Set the logo of the web page.
    image_path = "logo.png"
    st.image(image_path,
             caption='',
             use_column_width=True)

    st.title("Detección de relaciones causales en textos financieros")

    if "selected_example" not in st.session_state:
        st.session_state["selected_example"] = ""
    
    texto_financiero = st.text_area("",
                                    value=st.session_state["selected_example"],
                                    placeholder="Introduzca aquí un texto financiero",
                                    height=200,
                                    max_chars=1000)
    st.button("Predecir", type="primary")

    html_code_cause = ""
    html_code_effect = ""
    if len(texto_financiero) > 3:
        str_predicted_cause, str_predicted_effect = \
            predict_one_text(texto_financiero)

        causa_color = '<strong><span style="color: rgb(255, 165, 0);">causa</span></strong>'
        predicted_cause_color = '<strong><span style="color: rgb(255, 165, 0);">' + str_predicted_cause.strip() + '</span></strong>'
        html_code_cause = "La " + causa_color + " extraída\: :orange[" + predicted_cause_color + "]"
        st.markdown(html_code_cause, unsafe_allow_html=True)

        efecto_color = '<strong><span style="color: rgb(66, 115, 184);">efecto</span></strong>'
        predicted_effect_color = '<strong><span style="color: rgb(66, 115, 184);">' + str_predicted_effect.strip() + '</span></strong>'
        html_code_effect = "El " + efecto_color + " extraído\: :blue[" + predicted_effect_color + "]"
        st.markdown(html_code_effect, unsafe_allow_html=True)

    st.divider()

    st.write("Ejemplos de textos financieros:")

    ejemplo_1 = "Gracias a la crisis económica, " \
                "el banco A hace una fusión con el banco B"
    st.markdown("> " + ejemplo_1)
    if st.button("Predecir ejemplo nº1"):
        st.session_state["selected_example"] = ejemplo_1
        st.rerun()
    if st.session_state["selected_example"] == ejemplo_1:
        st.markdown(html_code_cause, unsafe_allow_html=True)
        st.markdown(html_code_effect, unsafe_allow_html=True)

    ejemplo_2 = "Las acciones tuvieron una gran recesión a " \
                "pesar del esfuerzo de los directores"
    st.markdown("> " + ejemplo_2)
    if st.button("Predecir ejemplo nº2"):
        st.session_state["selected_example"] = ejemplo_2
        st.rerun()
    if st.session_state["selected_example"] == ejemplo_2:
        st.markdown(html_code_cause, unsafe_allow_html=True)
        st.markdown(html_code_effect, unsafe_allow_html=True)

    ejemplo_3 = "Como la cultura de la empresa estaba tan " \
                "arraigada en los empleadores, ninguno fue " \
                "a la manifestación"
    st.markdown("> " + ejemplo_3)
    if st.button("Predecir ejemplo nº3"):
        st.session_state["selected_example"] = ejemplo_3
        st.rerun()
    if st.session_state["selected_example"] == ejemplo_3:
        st.markdown(html_code_cause, unsafe_allow_html=True)
        st.markdown(html_code_effect, unsafe_allow_html=True)

    ejemplo_4 = "Tuvimos el placer de designar a Paula " \
                "como su sustituta, no solo por su dilatada " \
                "experiencia en el sector del transporte y el " \
                "hecho de que ha sido consejera de la aerolínea, " \
                "sino también porque aporta otras perspectivas " \
                "de gran valor del mundo de las políticas públicas," \
                " la regulación y las empresas de suministros públicos. " \
                "Paula se incorporó en enero y, en nombre del " \
                "Consejo, deseo darle una calurosa bienvenida."
    st.markdown("> " + ejemplo_4)
    if st.button("Predecir ejemplo nº4"):
        st.session_state["selected_example"] = ejemplo_4
        st.rerun()
    if st.session_state["selected_example"] == ejemplo_4:
        st.markdown(html_code_cause, unsafe_allow_html=True)
        st.markdown(html_code_effect, unsafe_allow_html=True)

    st.divider()

    if st.checkbox("Mostrar más información sobre este proyecto"):
        st.write("El nombre completo del trabajo de fin de grado (TFG) es «Detección de relaciones "
                 "causales en documentos financieros empleando técnicas de procesamiento de lenguaje natural».")

        url = "https://github.com/JuncoDH/TFG_informatica"
        st.write("Esta aplicación web es la parte de explotación de [los resultados del TFG](%s)." % url)
        st.write("Se han implementado varias arquitecturas para resolver el problema.")
        st.write("Se han comparado luego todos los modelos y el que mejor resultados "
                 "ha dado es **tinyROBERTA**, con la arquitectura de pregunta-respuesta (QA, del inglés _question-answering_).")
        url = "https://huggingface.co/Juncodh/mi_tinyROBERTA_cause"
        st.write("Se puede utilizar el modelo para predecir [la causa](%s)" % url)
        url = "https://huggingface.co/Juncodh/mi_tinyROBERTA_effect"
        st.write("Se puede utilizar el modelo para predecir [el efecto](%s)" % url)

        st.write("")
        enter_key = '<span style="font-family: ' + "'Courier New'" + ', Courier, monospace;">enter</span>'
        st.markdown('<div style="background-color: #DDDDDD;">Para ejecutar el modelo solo hay '
                    'que introducir el texto financiero y presionar la tecla de %s.</div>' % enter_key, unsafe_allow_html=True)

        st.write("")
        st.markdown('<div style="background-color: #DDDDDD;">El texto financiero deberá tener de forma explícita la causa y el efecto.</div>\n', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div style="background-color: #DDDDDD;">El modelo normaliza los datos de salida de la causa y el efecto. '
                 'Borra algún carácter especial, los acentos, y convierte las mayúsculas a minúsculas.</div>', unsafe_allow_html=True)

        url1 = "https://wp.lancs.ac.uk/cfie/fincausal2023/"
        url2 = "http://www.lllf.uam.es/"
        st.write("")
        st.write("El conjunto de datos del entrenamiento y las métricas"
                 " utilizadas han sido extraídas de la competicion [FinCausal 2023](%s) "
                 "organizada por el [Laboratorio de Lingüística Informática](%s) de la Universidad de la UAM." % (url1, url2))


        st.write("**Autor**: Junco de las Heras Valenzuela")
        st.write("**Tutor**: Pablo Alfonso Haya Coll")

        url1 = "https://www.uam.es/EPS/Home.htm"
        url2 = "https://www.uam.es/uam/en/inicio"
        st.write("[Escuela Politécnica Superior](%s) (EPS) - [Universidad Autónoma de Madrid](%s) (UAM)" % (url1, url2))


if __name__ == "__main__":
    tokenizer, trainer_cause, trainer_effect = load_cache()
    main()
