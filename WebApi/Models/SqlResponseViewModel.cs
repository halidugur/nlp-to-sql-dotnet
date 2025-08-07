using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace stajprojem.Models
{
    [Table("products")]
    // Models/SqlResponseViewModel.cs
    public class SqlResponseViewModel
    {
        public bool success { get; set; }
        public string sql { get; set; }
        public string intent { get; set; }
        public string table { get; set; }
        public double? confidence { get; set; }
        public bool? has_time_filter { get; set; }
        public string error { get; set; }
        public double? elapsed { get; set; }

        // Bunlar sql sorgusunu veri tabanında sorgulatmamız ııcn gereklı
        public List<string>? Columns { get; set; }
        public List<Dictionary<string, object>>? Rows { get; set; }


    }


}
