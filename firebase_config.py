import pyrebase

const firebaseConfig = {
  apiKey: "AIzaSyA4AXSkS7Dhs134E2Kwc0UIW9FaDfSdCVE",
  authDomain: "sentiment-analysis-f6136.firebaseapp.com",
  projectId: "sentiment-analysis-f6136",
  storageBucket: "sentiment-analysis-f6136.firebasestorage.app",
  messagingSenderId: "455254508762",
  appId: "1:455254508762:web:18c3105e1fd244834b91e3",
  measurementId: "G-VM7KHC7WGV"
};

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
