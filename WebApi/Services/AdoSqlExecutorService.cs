using System.Data;
using Npgsql;

namespace stajprojem.Services
{
    public class AdoSqlExecutorService
    {
        private readonly string _connStr;

        public AdoSqlExecutorService(IConfiguration cfg)
        {
            _connStr = cfg.GetConnectionString("DefaultConnection");
        }

        public async Task<(List<string> cols, List<Dictionary<string, object>> rows)> ExecuteAsync(string sql)
        {
            var cols = new List<string>();
            var rows = new List<Dictionary<string, object>>();

            await using var conn = new NpgsqlConnection(_connStr);
            await conn.OpenAsync();

            await using var cmd = new NpgsqlCommand(sql, conn);
            await using var reader = await cmd.ExecuteReaderAsync(CommandBehavior.CloseConnection);

            // Kolon isimleri
            for (int i = 0; i < reader.FieldCount; i++)
                cols.Add(reader.GetName(i));

            // Satırlar
            while (await reader.ReadAsync())
            {
                var dict = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
                foreach (var c in cols)
                    dict[c] = reader[c];
                rows.Add(dict);
            }

            return (cols, rows);
        }
    }
}
