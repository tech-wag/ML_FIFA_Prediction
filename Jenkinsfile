pipeline {
  agent any

  environment {
    PYTHON = 'python3'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Install Dependencies') {
      steps {
        sh 'python -m pip install --upgrade pip'
        sh 'pip install -r requirements.txt'
      }
    }

    stage('Static Check') {
      steps {
        sh 'python -m py_compile DataLoader.py TrainModel.py Predictor.py app.py rag_enrich.py rag_production.py vertex_deploy.py'
      }
    }

    stage('Test') {
      steps {
        sh 'python -c "import pandas, sklearn, streamlit"'
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'IEEE_PAPER.pdf, *.py, requirements.txt, Dockerfile, Jenkinsfile', fingerprint: true
    }
  }
}
