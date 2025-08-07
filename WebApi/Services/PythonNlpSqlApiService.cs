using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace stajprojem.Services
{
    public class PythonNlpSqlApiService
    {
        private readonly HttpClient _httpClient;
        private readonly string _apiBaseUrl = "http://localhost:8000"; // FastAPI API adresi

        public PythonNlpSqlApiService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        // Türkçe doğal dil sorgusunu FastAPI'ye gönder, SQL cevabını al
        public async Task<string> GetSqlAsync(string naturalLanguageText)
        {
            var requestBody = new { text = naturalLanguageText };
            var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync($"{_apiBaseUrl}/generate-sql", content);
            response.EnsureSuccessStatusCode();

            var responseString = await response.Content.ReadAsStringAsync();
            dynamic result = JsonConvert.DeserializeObject(responseString);

            if (result.success == true)
                return result.sql;
            else
                return $"Hata: {result.error}";
        }
    }
}
